import tensorflow as tf
from tensorflow.keras import layers


class AnchorGenerator(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, 
             feature_maps, 
             strides, 
             base_sizes, 
             ratios, 
             scales, 
             image_sizes):
        
        ratios = tf.constant(ratios, dtype=tf.float32)
        scales = tf.constant(scales, dtype=tf.float32)

        if len(feature_maps) != len(strides) or \
           len(feature_maps) != len(base_sizes):
            raise ValueError("feature_maps, strides, and base_sizes must have the same length.")
        
        anchors = []
        for feature_map, stride, base_size in \
            zip(feature_maps, strides, base_sizes):
            
            anchors_fm = tf.map_fn(
                lambda args: self.anchors_per_feature_map(
                    args[0], args[1], ratios, scales, stride, 
                    base_size,
                ),
                elems=(feature_map, image_sizes),
                fn_output_signature=tf.TensorSpec(
                    shape=(None, 4), 
                    dtype=tf.float32
                ),
                parallel_iterations=32
            )
            # anchors_fm shape is [B, Ni, 4]
            anchors.append(anchors_fm)
        
        # anchors: list of [B, Ni, 4] -> [B, N, 4]
        anchors = tf.concat(anchors, axis=1)
        
        # tf.print('AnchorGenerator',
        #         'anchors :', tf.shape(anchors))
        return anchors
    
    def anchors_per_feature_map(
            self, 
            feature_map, 
            image_size, 
            ratios, 
            scales, 
            stride, 
            base_size):
        
        tf.print('feature_map:', feature_map, summarize=-1)
        tf.print('feature_map shape:', tf.shape(feature_map))

        # generate base anchors
        base_size = tf.cast(base_size, tf.float32)
        ratios = tf.reshape(ratios, [-1, 1]) # [3, 1]
        scales = tf.reshape(scales, [1, -1]) # [1, 3]

        area = ((base_size * scales) ** 2)
        tf.print('area:', area, summarize=-1)
        w_s = tf.sqrt(area)/ratios
        h_s = w_s * ratios
        w_s = tf.reshape(w_s, [-1])
        h_s = tf.reshape(h_s, [-1])
        x_1 = -w_s / 2
        y_1 = -h_s / 2
        x_2 = w_s / 2
        y_2 = h_s / 2
        # base_anchors shape is [A, 4]
        base_anchors = tf.stack([x_1, y_1, x_2, y_2], axis=1)
        tf.print('base_anchors:', base_anchors, summarize=-1)

        # shift base anchors to all locations on the feature map
        # shifts shape is [K, 4]
        height = tf.cast(tf.shape(feature_map)[0], tf.int32)
        width = tf.cast(tf.shape(feature_map)[1], tf.int32)
        shift_x = (tf.range(width, dtype=tf.float32) + 0.5) * stride
        shift_y = (tf.range(height, dtype=tf.float32) + 0.5) * stride
        shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
        shifts = tf.stack([
            tf.reshape(shift_x, [-1]),
            tf.reshape(shift_y, [-1]),
            tf.reshape(shift_x, [-1]),
            tf.reshape(shift_y, [-1])
        ], axis=1)
        # broadcast add, anchors shape is [K, A, 4]
        A = tf.shape(base_anchors)[0]
        K = tf.shape(shifts)[0]
        anchors = tf.reshape(base_anchors, [1, A, 4]) + \
            tf.reshape(shifts, [K, 1, 4])
        
        # reshape anchors shape to [K*A, 4]
        anchors = tf.reshape(anchors, [K * A, 4])

        # clip anchors to the image size
        # height = tf.cast(image_size[0], tf.float32)
        # width = tf.cast(image_size[1], tf.float32)
        # anchors = tf.stack([
        #     tf.clip_by_value(anchors[:, 0], 0, width - 1),
        #     tf.clip_by_value(anchors[:, 1], 0, height - 1),
        #     tf.clip_by_value(anchors[:, 2], 0, width - 1),
        #     tf.clip_by_value(anchors[:, 3], 0, height - 1)
        # ], axis=1)
        
        return anchors
    

if __name__ == '__main__':

    p2 = tf.fill((1, 10, 10, 1), value=0.0)
    
    # p3_0 = tf.fill((5, 5, 256), value=1.0)
    # p3_1 = tf.fill((5, 5, 256), value=1.1)
    # p3 = tf.stack([p3_0, p3_1], axis=0)

    # p4_0 = tf.fill((3, 3, 256), value=2.0)
    # p4_1 = tf.fill((3, 3, 256), value=2.1)
    # p4 = tf.stack([p4_0, p4_1], axis=0) 

    # p5_0 = tf.fill((2, 2, 256), value=3.0)
    # p5_1 = tf.fill((2, 2, 256), value=3.1)
    # p5 = tf.stack([p5_0, p5_1], axis=0) 

    tf.print('p2 shape:', tf.shape(p2))
    # tf.print('p3 shape:', tf.shape(p3))
    # tf.print('p4 shape:', tf.shape(p4))
    # tf.print('p5 shape:', tf.shape(p5))

    anchor_generator = AnchorGenerator()    
    anchors = anchor_generator(
        feature_maps=[ p2 ], 
        strides = [4],
        base_sizes = [32],
        ratios = [0.5, 1, 2],
        scales = [1.0, 1.5, 2.0],
        image_sizes = tf.constant(
            [[800, 800] ], 
            dtype=tf.int32
        )
    )
    
    tf.print('anchors shape:', tf.shape(anchors))
    for i in range(anchors.shape[1]):
        tf.print(i, ':', anchors[0, i], summarize=-1)
