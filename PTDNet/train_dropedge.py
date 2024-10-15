from config import *
import argparse
import tensorflow as tf
import time
from utils import *
from models import GCN_dropedge
from metrics import *

# def get_parm():
#     parser = argparse.ArgumentParser()

#     # settings
#     parser.add_argument("--dataset", type=str, default='cora', help="Dataset string")
#     parser.add_argument('--id', type=str, default='default_id', help='id to store in database')  #
#     parser.add_argument('--device', type=int, default=0,help='device to use')  #
#     parser.add_argument('--setting', type=str, default="description of hyper-parameters.")  #
#     parser.add_argument('--task_type', type=str, default='semi')
#     parser.add_argument('--early_stop', type=int, default= 100, help='early_stop')
#     parser.add_argument('--dtype', type=str, default='float32')  #
#     parser.add_argument('--seed',type=int, default=1234, help='seed')
#     parser.add_argument('--trails',type=int, default=5, help='trails')



#     # shared parameters
#     parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
#     parser.add_argument('--dropout',type=float, default=0.0, help='dropout rate (1 - keep probability).')
#     parser.add_argument('--weight_decay',type=float, default=5e-4, help='Weight for L2 loss on embedding matrix.')
#     parser.add_argument('--hiddens', type=str, default='256')
#     parser.add_argument("--lr", type=float, default=0.01,help='initial learning rate.')
#     parser.add_argument('--act', type=str, default='leaky_relu', help='activation funciton')  #
#     parser.add_argument('--initializer', default='he')
#     parser.add_argument('--L', type=int, default=1)  #
#     parser.add_argument('--outL', type=int, default=3)  #


#     # for dropedge
#     parser.add_argument('--dropedge',type=float, default=0.0, help='dropedge rate (1 - keep probability).')


#     # for PTDNet
#     parser.add_argument('--init_temperature', type=float, default=2.0)
#     parser.add_argument('--temperature_decay', type=float, default=0.99)
#     parser.add_argument('--denoise_hidden_1', type=int, default=16)
#     parser.add_argument('--denoise_hidden_2', type=int, default=0)
#     #
#     parser.add_argument('--gamma', type=float, default=-0.0)
#     parser.add_argument('--zeta', type=float, default=1.01)

#     parser.add_argument('--lambda1', type=float, default=0.1, help='Weight for L0 loss on laplacian matrix.')
#     parser.add_argument('--lambda3', type=float, default=0.01, help='Weight for nuclear loss')
#     parser.add_argument("--coff_consis", type=float, default=0.01,help='consistency')
#     parser.add_argument('--k_svd', type=int, default=1)
#     args, _ = parser.parse_known_args()
#     return args

# # Settings
# args = get_parm()

# Settings
optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(args.dataset)

tuple_adj = sparse_to_tuple(adj.tocoo())
adj_tensor = tf.SparseTensor(*tuple_adj)

features = preprocess_features(features)
import time
begin = time.time()

model = GCN_dropedge(input_dim=features.shape[1], output_dim=y_train.shape[1], adj=adj_tensor)


features_tensor = tf.convert_to_tensor(features,dtype=tf.float32)
y_train_tensor = tf.convert_to_tensor(y_train,dtype=tf.float32)
train_mask_tensor = tf.convert_to_tensor(train_mask)
y_test_tensor = tf.convert_to_tensor(y_test,dtype=tf.float32)
test_mask_tensor = tf.convert_to_tensor(test_mask)
y_val_tensor = tf.convert_to_tensor(y_val,dtype=tf.float32)
val_mask_tensor = tf.convert_to_tensor(val_mask)
print(y_train_tensor.get_shape().as_list(),y_test_tensor.get_shape().as_list(),y_val_tensor.get_shape().as_list())
best_test_acc = 0
best_val_acc = 0
best_val_loss = 10000


curr_step = 0
for epoch in range(args.epochs):

    with tf.GradientTape() as tape:
        output = model.call((features_tensor),training=True)
        cross_loss = masked_softmax_cross_entropy(output, y_train_tensor,train_mask_tensor)
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables])
        loss = cross_loss + args.weight_decay*lossL2
        grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    output = model.call((features_tensor), training=False)
    train_acc = masked_accuracy(output, y_train_tensor,train_mask_tensor)
    val_acc  = masked_accuracy(output, y_val_tensor,val_mask_tensor)
    val_loss = masked_softmax_cross_entropy(output, y_val_tensor, val_mask_tensor)
    test_acc  = masked_accuracy(output, y_test_tensor,test_mask_tensor)

    if val_acc > best_val_acc:
        curr_step = 0
        best_test_acc = test_acc
        best_val_acc = val_acc
        best_val_loss= val_loss
        # Print results

    else:
        curr_step +=1
    if curr_step > args.early_stop:
        print("Early stopping...")
        break

    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cross_loss),"val_loss=", "{:.5f}".format(val_loss),
      "train_acc=", "{:.5f}".format(val_acc), "val_acc=", "{:.5f}".format(val_acc),
      "test_acc=", "{:.5f}".format(best_test_acc))
    end = time.time()
    print('time ',(end-begin))