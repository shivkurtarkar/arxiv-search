import os
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from tensorboard.plugins import projector

LOG_DIR = './logs/'

INDEX_NAME = 'doc'
LABEL_FILE=f'{INDEX_NAME}_labels.tsv'
VECTOR_FILE=f'{INDEX_NAME}_vectors.tsv'
FEATURE_FILE=f'{INDEX_NAME}_vectors.txt'
EMBEDDING_CHECKPOINT='embedding.ckpt'

feature_vectors = np.loadtxt(os.path.join(LOG_DIR,FEATURE_FILE))
weights = tf.Variable(feature_vectors)

# create checkpoint
checkpoint = tf.train.Checkpoint(embedding=weights)
checkpoint.save(os.path.join(LOG_DIR, EMBEDDING_CHECKPOINT))

# setup config
config = projector.ProjectorConfig()
embedding = config.embeddings.add()
# The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`
embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
embedding.metadata_path = LABEL_FILE
projector.visualize_embeddings(LOG_DIR, config)