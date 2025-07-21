dataset_GAN.py - Сonjunto de datos personalizado con las imagenes de textil no dañado y sin anomalias para el aprendizaje de los modelos UNetGenerator y PatchDiscriminator de GAN.

patch_discriminator_chatbot_GAN.py - modelo PatchDiscriminator de GAN.

unet_generator_chatbot_GAN.py - modelo UNetGenerator de GAN.

train_GAN.py - File de aprendizaje de los modelos UNetGenerator y PatchDiscriminator de GAN para la deteccion de anomalias y defectos de textil en las imagenes.

FastAPI - Chatbot en WhatsApp con las respuestas de texto con indicaciones para las respuestas generadas por el modelo gpt-3.5-turbo de OpenAI con el registro y la verificacion de los usuarios de chatbot y implementación del proyecto usando Docker Compose. Adicionalmente, con la posibilidad de generacion de heatmaps generadas por los modelos UNetGenerator y PatchDiscriminator en respuesta a una imagen de entrada y guardandolas en la base de datos (usando por ejemplo PostgreSQL).

