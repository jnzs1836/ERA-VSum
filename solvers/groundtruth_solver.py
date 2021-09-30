import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
from .gan_solver import GANSolver


class GroundTruthSolver(GANSolver):
    def __init__(self, config):
        super(GroundTruthSolver, self).__init__(config)
        self.lambda_gp = 0.5
        self.lambda_l2 = 20
        self.feature_l2_loss_weight = 1
        self.with_images = config.with_images

    def feature_l2_loss(self, fake_features, real_features):
        loss = (fake_features - real_features).norm(dim=2).mean(dim=0)
        return loss

    def gradiant_penalty_loss(self, generated_features, real_features):
        batch_size = 1
        eta = torch.FloatTensor(1, 1, 1).uniform_(0, 1)
        eta = eta.expand(real_features.size(0), real_features.size(1), real_features.size(2))

        eta = eta.cuda()
        interpolated = eta * real_features + ((1 - eta) * generated_features)

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        _, prob_interpolated = self.discriminator(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                  grad_outputs=torch.ones(
                                      prob_interpolated.size()).cuda(),
                                  create_graph=True, retain_graph=True)[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_gp
        return grad_penalty

    def l2_distance_loss(self, fake_prob, real_prob):
        l2 = (fake_prob - real_prob).norm()
        return l2 * self.lambda_l2

    def variance_loss(self, scores, epsilon=1e-4):
        median_tensor = torch.zeros(scores.shape[0]).to(self.device)
        median_tensor.fill_(torch.median(scores))
        loss = nn.MSELoss()
        variance = loss(scores.squeeze(), median_tensor)
        return 1 / (variance + epsilon)

    def step(self, batch_i, batch, step_type="train"):
        image_features = batch[0]
        image_features = image_features.view(-1, self.input_size)
        image_features_ = Variable(image_features).cuda()

        video_key = batch[-2][0]

        # ---- Train sLSTM, eLSTM ----#
        # if self.config.verbose:
        #     tqdm.write('\nTraining sLSTM and eLSTM...')

        # [seq_len, 1, hidden_size]
        feature_l2_losses = []
        if self.with_images:
            images = batch[-1]
        else:
            images = None

        target = batch[7]
        target = target.cuda()
        # target = dataset['gtscore'][...]
        # target = torch.from_numpy(target).unsqueeze(0)
        # target = target.squeeze(0)
        # Normalize frame scores
        target -= target.min()
        target /= target.max()
        self.s_e_optimizer.zero_grad()

        original_features = self.linear_compress(image_features_.detach()).unsqueeze(1)
        noise = self.generate_noise(original_features.size())
        scores, h_mu, h_log_variance, generated_features = self.summarizer(
            original_features, z=noise, images=images, video_key=video_key, target=target)
        _, _, _, uniform_features = self.summarizer(
            original_features, uniform=True, images=images, video_key=video_key, target=target)
        feature_l2_loss = self.feature_l2_loss(generated_features, original_features)
        feature_l2_losses.append(feature_l2_loss)

        feature_l2_losses = torch.stack(feature_l2_losses, dim=1)
        min_feature_l2_losses = torch.min(feature_l2_losses, dim=1)
        min_feature_l2_losses = min_feature_l2_losses[0][0]
        h_origin, original_prob = self.discriminator(original_features)
        h_fake, fake_prob = self.discriminator(generated_features)
        h_uniform, uniform_prob = self.discriminator(uniform_features)
        reconstruction_loss = self.reconstruction_loss(h_origin, h_fake)
        prior_loss = self.prior_loss(h_mu, h_log_variance)
        sparsity_loss = self.sparsity_loss(scores)
        variance_loss = self.variance_loss(scores, 1e-4)
        # print(variance_loss)
        # print(reconstruction_loss)
        # print(scores)
        s_e_loss = reconstruction_loss + prior_loss + sparsity_loss + variance_loss
        # s_e_loss = prior_loss
        if step_type == "train":
            s_e_loss.backward()  # retain_graph=True)
            # Gradient cliping
            torch.nn.utils.clip_grad_norm(self.model.parameters(), self.config.clip)
            self.s_e_optimizer.step()
            # self.s_e_optimizer.zero_grad()

        # ---- Train dLSTM ----#
        # if self.config.verbose:
        #     tqdm.write('Training dLSTM...')

        # [seq_len, 1, hidden_size]
        self.d_optimizer.zero_grad()
        self.s_e_optimizer.zero_grad()
        self.c_optimizer.zero_grad()
        original_features = self.linear_compress(image_features_.detach()).unsqueeze(1)
        noise = self.generate_noise(original_features.size())
        scores, h_mu, h_log_variance, generated_features = self.summarizer(
            original_features,
            z=noise,
            images=images,
            video_key=video_key,
            target=target,
        )
        _, _, _, uniform_features = self.summarizer(
            original_features, uniform=True, images=images, video_key=video_key, target=target
        )

        h_origin, original_prob = self.discriminator(original_features)
        h_fake, fake_prob = self.discriminator(generated_features)
        h_uniform, uniform_prob = self.discriminator(uniform_features)

        reconstruction_loss = self.reconstruction_loss(h_origin, h_fake)

        gan_loss = self.l2_distance_loss(fake_prob, original_prob)
        d_loss = reconstruction_loss + gan_loss

        if step_type == "train":
            d_loss.backward()  # retain_graph=True)
            # Gradient cliping
            torch.nn.utils.clip_grad_norm(self.model.parameters(), self.config.clip)
            self.d_optimizer.step()
            # self.d_optimizer.zero_grad()

        c_loss = 0
        # ---- Train cLSTM ----#
        # if batch_i > self.config.discriminator_slow_start or True:
        with torch.backends.cudnn.flags(enabled=False):
            # [seq_len, 1, hidden_size]
            self.c_optimizer.zero_grad()
            self.s_e_optimizer.zero_grad()
            image_features_ = Variable(image_features).cuda()
            original_features = self.linear_compress(image_features_.detach()).unsqueeze(1)
            noise = self.generate_noise(original_features.size())
            scores, h_mu, h_log_variance, generated_features = self.summarizer(
                original_features, z=noise, images=images, video_key=video_key, target=target)
            _, _, _, uniform_features = self.summarizer(
                original_features, uniform=True, images=images, video_key=video_key, target=target)

            h_origin, original_prob = self.discriminator(original_features)
            h_fake, fake_prob = self.discriminator(generated_features)
            h_uniform, uniform_prob = self.discriminator(uniform_features)
            # Maximization

            gp_loss = self.gradiant_penalty_loss(generated_features, original_features)
            c_loss = -1 * self.l2_distance_loss(fake_prob, original_prob) + gp_loss
            if step_type == "train":
                c_loss.backward()
                # for p in self.linear_compress.parameters():
                #    print(p.grad)
                # Gradient cliping
                torch.nn.utils.clip_grad_norm(self.model.parameters(), self.config.clip)
                self.c_optimizer.step()
                # self.c_optimizer.zero_grad()
        probs = {
            "original_prob": original_prob.mean().data,
            "fake_prob": fake_prob.mean().data,
            "uniform_prob": uniform_prob.mean().data
        }
        losses = {
            "sparsity_loss": sparsity_loss.data,
            "prior_loss": prior_loss.data,
            "gan_loss": gan_loss.data,
            "recon_loss": reconstruction_loss.data,
            "s_e_loss": s_e_loss.data,
            "d_loss": d_loss.data,
            "c_loss": c_loss
        }
        metrics = self.evaluate_results(scores, losses, probs, batch)
        if step_type == "train":
            self.s_e_scheduler.step()
            self.c_scheduler.step()
            self.d_scheduler.step()
        return scores, losses, probs, metrics

    def valid_step(self, batch_i, batch):
        self.model.eval()
        return self.step(batch_i, batch, "valid")





