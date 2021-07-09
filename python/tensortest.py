import tensorflow as tf
import argparse
import logging
from time import perf_counter

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--number', dest='number', action='store', type=int,
                    default=1024,
                    help='Optional, integer')
parser.add_argument('--dtype', dest='dtype', action='store', type=str,
                    default='float32',
                    choices=['float64','float32','float16','bfloat16','int64','int32','int16','int8','uint64','uint32','uint16','uint8'],
                    help='Optional, dtype')

args = parser.parse_args()

size = args.number;

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

tf.debugging.set_log_device_placement(True)

logging.info("Generating a %dx%d randomUniform matrix, please standby", size, size);
t1_start = perf_counter()
matrix = tf.random_uniform_initializer(minval=0, maxval=1, seed=2)(shape=[size,size],dtype=args.dtype);
t1_stop = perf_counter()
print("Generating a ",size,"x",size," matrix took: ", (t1_stop - t1_start), 's')

# Initialize cuBLAS....
temp = tf.constant([0.0,1.0,2.0,3.0],shape=[2,2],dtype="float32");
temp2 = tf.matmul(temp,temp,transpose_a=False,transpose_b=True);

t1_start = perf_counter()
result = tf.matmul(matrix,matrix,transpose_a=False,transpose_b=True);
t1_stop = perf_counter()
print("Multiplying a ",size,"x",size," matrix took: ", (t1_stop - t1_start), 's')