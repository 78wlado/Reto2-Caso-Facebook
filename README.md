# Reto2-Caso-Facebook
Caso Facebook: Fake news, desinformación y redes neuronales

# Localiza fake news en la prensa o en redes sociales. ¿Cómo sabes que son fake news?

Existen algunas estrategias que puedes utilizar para evaluar la veracidad de una noticia o información: Verifica la fuente, busca múltiples fuentes, confirma los hechos, consulta verificadores de hechos, tener en cuenta las fechas y los contextos.

# Localiza un ejemplo de deepfake cuya intención sea la de engañar o hacer daño.

 https://www.wsj.com/articles/fraudsters-use-ai-to-mimic-ceos-voice-in-unusual-cybercrime-case-11567157402

# ¿Serías capaz de explicar cómo funciona la tecnología de deep learning para generar deepfakes?

La tecnología de deep learning se utiliza en la generación de deepfakes para crear contenido falso convincente mediante la manipulación de imágenes o videos. Aquí se explica de manera general cómo funciona la tecnología de deep learning en el proceso de generación de deepfakes:

1.	Recopilación de datos.
2.	Entrenamiento de la red neuronal.
3.	Extracción de características.
4.	Generación de la imagen falsa.
5.	Refinamiento y mejora.

# Busca información sobre los ataques adversarios contra redes neuronales.
Estos ataques pueden tener implicaciones de seguridad y privacidad, y han sido objeto de una investigación considerable en el campo del aprendizaje automático. Aquí hay algunos conceptos clave relacionados con los ataques adversarios:
![image](https://github.com/78wlado/Reto2-Caso-Facebook/assets/136178520/b42427be-3eda-4782-8598-21b3f8155db0)

# Codigo

Importar las bibliotecas necesarias:

    import tensorflow as tf
    from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Reshape, Conv2DTranspose
    from tensorflow.keras.models import Model

Definir la arquitectura de la red generativa (generador) y la red discriminativa (discriminador):

    # Generador
    def build_generator():
        # Definir la estructura de la red generativa
        generator_input = Input(shape=(latent_dim,))
        # Capas convolucionales, reshape, etc.
        # ...
        # Capa de salida
        generator_output = Conv2DTranspose(3, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='tanh')(generator_layers)
        generator = Model(generator_input, generator_output)
        return generator
    # Discriminador
    def build_discriminator():
        # Definir la estructura de la red discriminativa
        discriminator_input = Input(shape=(image_width, image_height, image_channels))
        # Capas convolucionales, flatten, dense, etc.
        # ...
        # Capa de salida
        discriminator_output = Dense(1, activation='sigmoid')(discriminator_layers)
        discriminator = Model(discriminator_input, discriminator_output)
        return discriminator

Definir la función de pérdida y los optimizadores:

        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        
        def discriminator_loss(real_output, fake_output):
            # Definir la función de pérdida del discriminador
            real_loss = loss(tf.ones_like(real_output), real_output)
            fake_loss = loss(tf.zeros_like(fake_output), fake_output)
            total_loss = real_loss + fake_loss
            return total_loss
        
        def generator_loss(fake_output):
            # Definir la función de pérdida del generador
            return loss(tf.ones_like(fake_output), fake_output)
        
        # Optimizadores
        generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

Definir el bucle de entrenamiento para entrenar la GAN:
        def train_step(images):
        
            # Generar ruido aleatorio como entrada para el generador
            noise = tf.random.normal([batch_size, latent_dim])
            
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                # Generar imágenes falsas con el generador
                generated_images = generator(noise, training=True)
        
                # Calcular las salidas del discriminador para las imágenes reales y falsas
                real_output = discriminator(images, training=True)
                fake_output = discriminator(generated_images, training=True)
        
                # Calcular las pérdidas del generador y el discriminador
                gen_loss = generator_loss(fake_output)
                disc_loss = discriminator_loss(real_output, fake_output)
        
            # Calcular los gradientes y aplicarlos mediante los optimizadores
            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient





