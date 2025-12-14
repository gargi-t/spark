import numpy as np  
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (Dense, Input, Dropout, BatchNormalization, 
                                    concatenate, GaussianNoise, LeakyReLU)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import keras_tuner as kt

class BayesianDenseLayer(tf.keras.layers.Layer):
    """Bayesian dense layer with uncertainty estimation"""
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        
    def build(self, input_shape):
        self.kernel_mu = self.add_weight(
            name='kernel_mu',
            shape=(input_shape[-1], self.units),
            initializer='glorot_normal'
        )
        self.kernel_rho = self.add_weight(
            name='kernel_rho',
            shape=(input_shape[-1], self.units),
            initializer='zeros'
        )
        self.bias_mu = self.add_weight(
            name='bias_mu',
            shape=(self.units,),
            initializer='zeros'
        )
        self.bias_rho = self.add_weight(
            name='bias_rho',
            shape=(self.units,),
            initializer='zeros'
        )
        
    def call(self, inputs, training=False):
        if training:
            # Sample from variational distribution
            kernel_sigma = tf.math.softplus(self.kernel_rho)
            kernel = self.kernel_mu + kernel_sigma * tf.random.normal(self.kernel_mu.shape)
            
            bias_sigma = tf.math.softplus(self.bias_rho)
            bias = self.bias_mu + bias_sigma * tf.random.normal(self.bias_mu.shape)
            
            return tf.matmul(inputs, kernel) + bias
        else:
            # Use mean values for prediction
            return tf.matmul(inputs, self.kernel_mu) + self.bias_mu

class SpectralRetrievalNN:
    """Advanced neural network for atmospheric retrieval with uncertainty"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.history = None
        
    def build_model(self, input_dim, output_dim, bayesian=False):
        """Build advanced neural network architecture"""
        inputs = Input(shape=(input_dim,))
        
        # Add Gaussian noise for regularization
        x = GaussianNoise(0.001)(inputs)
        
        # Create multiple parallel processing branches
        branches = []
        for _ in range(3):  # Three different filter sizes
            branch = Dense(128, activation='relu')(x)
            branch = BatchNormalization()(branch)
            branch = Dropout(self.config['model']['dropout_rate'])(branch)
            branches.append(branch)
        
        # Concatenate and process further
        x = concatenate(branches)
        
        # Main processing layers
        for units in self.config['model']['nn_layers']:
            if bayesian:
                x = BayesianDenseLayer(units)(x)
                x = LeakyReLU(alpha=0.1)(x)
            else:
                x = Dense(units, activation='relu', 
                         kernel_regularizer=l2(0.001))(x)
                x = BatchNormalization()(x)
                x = Dropout(self.config['model']['dropout_rate'])(x)
        
        # Output layer with uncertainty estimation
        if bayesian:
            outputs = BayesianDenseLayer(output_dim)(x)
        else:
            # Standard output with mean and variance
            mean_output = Dense(output_dim, name='mean_output')(x)
            log_var_output = Dense(output_dim, name='log_var_output')(x)
            outputs = concatenate([mean_output, log_var_output])
        
        self.model = Model(inputs=inputs, outputs=outputs)
        
        # Custom loss for uncertainty estimation
        def heteroscedastic_loss(y_true, y_pred):
            if bayesian:
                return tf.reduce_mean(tf.square(y_true - y_pred))
            else:
                mean = y_pred[:, :output_dim]
                log_var = y_pred[:, output_dim:]
                precision = tf.exp(-log_var)
                return tf.reduce_mean(0.5 * (precision * tf.square(y_true - mean) + log_var))
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config['model']['learning_rate'])
        self.model.compile(optimizer=optimizer, loss=heteroscedastic_loss,
                          metrics=['mae', 'mse'])
        
        print(self.model.summary())
        return self.model
    
    def hyperparameter_tuning(self, X_train, y_train, X_val, y_val):
        """Perform hyperparameter optimization using Keras Tuner"""
        
        def build_model(hp):
            model = tf.keras.Sequential()
            model.add(Input(shape=(X_train.shape[1],)))
            
            # Tune number of layers
            for i in range(hp.Int('num_layers', 2, 6)):
                model.add(Dense(units=hp.Int(f'units_{i}', 32, 512, step=32),
                               activation=hp.Choice(f'activation_{i}', ['relu', 'tanh', 'elu'])))
                model.add(Dropout(hp.Float(f'dropout_{i}', 0.1, 0.5)))
            
            model.add(Dense(y_train.shape[1]))
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
                loss='mse',
                metrics=['mae']
            )
            return model
        
        tuner = kt.BayesianOptimization(
            build_model,
            objective='val_mae',
            max_trials=20,
            executions_per_trial=2,
            directory='hyperparameter_tuning',
            project_name='inara_nn'
        )
        
        tuner.search(X_train, y_train,
                    epochs=50,
                    validation_data=(X_val, y_val),
                    callbacks=[EarlyStopping(patience=10)])
        
        # Get optimal hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(f"Best hyperparameters: {best_hps.values}")
        
        return tuner.get_best_models(num_models=1)[0]
    
    def train(self, X_train, y_train, X_val, y_val, epochs=200):
        """Train the neural network with advanced callbacks"""
        checkpoint = ModelCheckpoint(
            'best_model.h5', monitor='val_loss', save_best_only=True, verbose=1
        )
        early_stop = EarlyStopping(monitor='val_loss', patience=self.config['model']['patience'], 
                                  restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7)
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=self.config['data']['batch_size'],
            validation_data=(X_val, y_val),
            callbacks=[checkpoint, early_stop, reduce_lr],
            verbose=1
        )
        
        return self.history
    
    def predict_with_uncertainty(self, X, n_samples=100):
        """Make predictions with uncertainty estimation"""
        if any('Bayesian' in layer.__class__.__name__ for layer in self.model.layers):
            # Bayesian model: sample multiple times
            predictions = np.array([self.model.predict(X, verbose=0) for _ in range(n_samples)])
            means = np.mean(predictions, axis=0)
            stds = np.std(predictions, axis=0)
            return means, stds
        else:
            # Standard model with variance output
            pred = self.model.predict(X, verbose=0)
            means = pred[:, :pred.shape[1]//2]
            log_vars = pred[:, pred.shape[1]//2:]
            stds = np.exp(0.5 * log_vars)
            return means, stds