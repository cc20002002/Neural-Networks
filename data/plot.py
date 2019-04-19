from __future__ import absolute_import, division, print_function
import tensorflow as tf
import h5py
import numpy as np
mnist = tf.keras.datasets.fashion_mnist

with h5py.File('train_128.h5','r') as H:
  data = np.copy(H['data'])
with h5py.File('train_label.h5','r') as H:
  label = np.copy(H['label'])
with h5py.File('test_128.h5','r') as H:
  test = np.copy(H['data'])
train_128Train=data[0:60000,:]
train_128Test=test
train_labelTrain=label[0:60000]
train_labelTest=label[50000:60000]
(x_train, y_train), (x_test, y_test) = mnist.load_data()
train_labelTest = y_test
max(train_labelTrain)
np.random.seed(3)
tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
model = tf.keras.models.Sequential([  
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Dense(160, activation=tf.nn.tanh, kernel_regularizer=tf.keras.regularizers.l2(0.0007)),   # continue search 0.0015 --0.004 
  tf.keras.layers.Dropout(0.05),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Dense(160, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.0007)),   # continue search 0.0015 --0.004
  tf.keras.layers.Dropout(0.05),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.11,momentum=0.9),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_128Train, train_labelTrain, epochs=44, batch_size=1500, callbacks=[tbCallBack])
model.evaluate(train_128Test, train_labelTest)
from keras.utils import plot_model
plot_model(model, to_file='model.eps')


from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import numpy as np
np.random.seed(0)

# parameter ==========================
wkdir = './'
pb_filename = 'model.pb'


# save model to pb ====================
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.
    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

# save keras model as tf pb files ===============
from keras import backend as K
frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in model.outputs])
tf.train.write_graph(frozen_graph, wkdir, pb_filename, as_text=False)


# # load & inference the model ==================

from tensorflow.python.platform import gfile
with tf.Session() as sess:
    # load model from pb file
    with gfile.FastGFile(wkdir+'/'+pb_filename,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        g_in = tf.import_graph_def(graph_def)
    # write to tensorboard (check tensorboard for each op names)
    writer = tf.summary.FileWriter(wkdir+'/log/')
    writer.add_graph(sess.graph)
    writer.flush()
    writer.close()
    # print all operation names 
    print('\n===== ouptut operation names =====\n')
    for op in sess.graph.get_operations():
      print(op)
    # inference by the model (op name must comes with :0 to specify the index of its output)
    tensor_output = sess.graph.get_tensor_by_name('import/dense_3/Sigmoid:0')
    tensor_input = sess.graph.get_tensor_by_name('import/dense_1_input:0')
    predictions = sess.run(tensor_output, {tensor_input: x})
    print('\n===== output predicted results =====\n')
    print(predictions)