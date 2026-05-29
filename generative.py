from tensorflow.keras.layers import Input, Dense, Lambda, Reshape, Conv1D, Flatten  # Added Input, Dense
from tensorflow.keras.models import Model  # Added this line
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np  # Add this too

class EnhancedAtmosphericVAE:
    """Enhanced VAE with physical constraints"""
    
    def __init__(self, config):
        self.config = config
        self.encoder = None
        self.decoder = None
        self.vae = None
        self.original_dim = None
        
    def build_model(self, original_dim):
        """Build enhanced VAE with physical constraints"""
        self.original_dim = original_dim
        latent_dim = self.config['model']['latent_dim']
        
        # Encoder
        encoder_inputs = Input(shape=(original_dim,))
        x = Dense(512, activation='relu')(encoder_inputs)
        x = Dense(256, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        
        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)
        
        # Sampling layer
        def sampling(args):
            z_mean, z_log_var = args
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        
        z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
        
        self.encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
        
        # Decoder
        latent_inputs = Input(shape=(latent_dim,))
        x = Dense(128, activation='relu')(latent_inputs)
        x = Dense(256, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        decoder_outputs = Dense(original_dim, activation='linear')(x)
        
        self.decoder = Model(latent_inputs, decoder_outputs, name='decoder')
        
        # VAE model
        outputs = self.decoder(z)
        self.vae = Model(encoder_inputs, outputs, name='vae')
        
        # Physical constraint loss
        def physical_constraint_loss(params):
            """Apply physical constraints to generated parameters"""
            # Ensure positive abundances
            abundance_loss = tf.reduce_mean(tf.square(tf.nn.relu(-params[:, :8])))
            
            # Temperature decreasing with pressure
            t_gradient_loss = tf.reduce_mean(tf.square(tf.nn.relu(
                params[:, 8:9] - params[:, 9:10])))  # T_day should be >= T_night
            
            return self.config['model']['vae_beta'] * (abundance_loss + t_gradient_loss)
        
        # VAE loss
        reconstruction_loss = tf.keras.losses.mse(encoder_inputs, outputs)
        reconstruction_loss *= original_dim
        
        kl_loss = 1 + z_log_var - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_log_var)
        kl_loss = tf.keras.backend.mean(kl_loss)
        kl_loss *= -0.5
        
        physics_loss = physical_constraint_loss(outputs)
        
        vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss + physics_loss)
        
        self.vae.add_loss(vae_loss)
        self.vae.compile(optimizer='adam')
        
    def train(self, X_train, epochs=100):
        """Train the enhanced VAE"""
        self.vae.fit(X_train, epochs=epochs, batch_size=self.config['data']['batch_size'], verbose=1)
        
    def generate_samples(self, n_samples=100):
        """Generate samples from the latent space"""
        z_samples = np.random.normal(0, 1, (n_samples, self.config['model']['latent_dim']))
        return self.decoder.predict(z_samples, verbose=0)
    
    def interpolate_in_latent_space(self, params1, params2, n_steps=10):
        """Interpolate between two parameter sets in latent space"""
        z1_mean, _, _ = self.encoder.predict(params1.reshape(1, -1), verbose=0)
        z2_mean, _, _ = self.encoder.predict(params2.reshape(1, -1), verbose=0)
        
        interpolated_params = []
        for alpha in np.linspace(0, 1, n_steps):
            z = alpha * z1_mean + (1 - alpha) * z2_mean
            interpolated_params.append(self.decoder.predict(z, verbose=0))
            
        return np.vstack(interpolated_params)

class AtmosphericGAN:
    """GAN for generating realistic atmospheric parameters"""
    
    def __init__(self, config):
        self.config = config
        self.generator = None
        self.discriminator = None
        self.gan = None
        
    def build_generator(self):
        """Build generator network"""
        noise_dim = self.config['model']['gan_noise_dim']
        input_noise = Input(shape=(noise_dim,))
        
        x = Dense(128, activation='relu')(input_noise)
        x = BatchNormalization()(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        
        # Output layer with physical constraints
        outputs = Dense(self.config['data']['n_molecular_species'] + 6,  # molecules + T + P + clouds
                       activation='linear')(x)
        
        self.generator = Model(input_noise, outputs)
        return self.generator
    
    def build_discriminator(self):
        """Build discriminator network"""
        input_params = Input(shape=(self.config['data']['n_molecular_species'] + 6,))
        
        x = Dense(512, activation='relu')(input_params)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        
        validity = Dense(1, activation='sigmoid')(x)
        
        self.discriminator = Model(input_params, validity)
        self.discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return self.discriminator
    
    def build_gan(self):
        """Build complete GAN"""
        self.discriminator.trainable = False
        gan_input = Input(shape=(self.config['model']['gan_noise_dim'],))
        generated_params = self.generator(gan_input)
        validity = self.discriminator(generated_params)
        
        self.gan = Model(gan_input, validity)
        self.gan.compile(loss='binary_crossentropy', optimizer='adam')
        return self.gan
    
    def train(self, real_params, epochs=10000, batch_size=32):
        """Train the GAN"""
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        for epoch in range(epochs):
            # Train discriminator
            idx = np.random.randint(0, real_params.shape[0], batch_size)
            real_samples = real_params[idx]
            
            noise = np.random.normal(0, 1, (batch_size, self.config['model']['gan_noise_dim']))
            fake_samples = self.generator.predict(noise, verbose=0)
            
            d_loss_real = self.discriminator.train_on_batch(real_samples, valid)
            d_loss_fake = self.discriminator.train_on_batch(fake_samples, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # Train generator
            noise = np.random.normal(0, 1, (batch_size, self.config['model']['gan_noise_dim']))
            g_loss = self.gan.train_on_batch(noise, valid)
            
            if epoch % 1000 == 0:
                print(f"{epoch} [D loss: {d_loss[0]} | D accuracy: {100*d_loss[1]}] [G loss: {g_loss}]")
    
    def generate_samples(self, n_samples=100):
        """Generate samples using trained generator"""
        noise = np.random.normal(0, 1, (n_samples, self.config['model']['gan_noise_dim']))
        return self.generator.predict(noise, verbose=0)
