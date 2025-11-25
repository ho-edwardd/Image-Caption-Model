# CLIP-Captioning-Model

This project implements an image captioning system built upon two state-of-the-art models: **CLIP** (Contrastive Language–Image Pretraining) and **GPT-2** (Generative Pre-trained Transformer 2). The system leverages CLIP's robust vision-language understanding and GPT-2's powerful generative capabilities to produce accurate, contextually relevant descriptions. Crucially, a lightweight Transformer architecture translates the encoded CLIP visual features, ensuring they share the same semantic space as the GPT embeddings without requiring any fine-tuning of the base models.

Sample outputs demonstrating the system's performance can be found in the examples folder, which contains predicted captions for five randomly selected images from the 2017 MSCOCO testing dataset.

The core contribution of this project is the implementation of a Transformer-based mapping network. This network bridges the CLIP and GPT-2 pre-trained models via a series of image-to-prefix embedding translations. This architecture offers several benefits:  
- **Efficiency**: It significantly reduces the number of trainable parameters, leading to faster training times and more efficient computation.  
- **Performance**: The model achieves performance equal to or better than traditional methods across three different test sets.  
- **Frozen Pre-Trained Models**: The internal architecture allows for a robust image-to-caption system without altering the weights of the two massive pre-trained models. This avoids the need to retrain the GPT network or fine-tune internal layers, which are common bottlenecks in traditional targeted learning systems.