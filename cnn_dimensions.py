import tensorflow as tf

input = tf.placeholder(tf.float32, (None, 32, 32, 3))
filter_weights = tf.Variable(tf.truncated_normal((8, 8, 3, 20))) # (height, width, input_depth, output_depth)
filter_bias = tf.Variable(tf.zeros(20))
strides = [1, 2, 2, 1] # (batch, height, width, depth)
padding = 'VALID'
conv = tf.nn.conv2d(input, filter_weights, strides, padding) + filter_bias
print(conv)

#Note the output shape of conv will be [1, 13, 13, 20]. It's 4D to account for batch size, but more importantly, i
# t's not [1, 14, 14, 20]. This is because the padding algorithm TensorFlow uses is not exactly the same as the one above.
# An alternative algorithm is to switch padding from 'VALID' to SAME which would result in an output shape of [1, 16, 16, 20].