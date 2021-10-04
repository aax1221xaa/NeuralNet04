import tensorflow as tf


DEBUG = 1

class MultiBoxLoss:
    def __init__(self, C):
        self.alpha = C.alpha
        self.neg_pos_ratio = C.neg_pos_ratio

    def L1_smooth_loss(self, y_true, y_pred):
        abs_val = tf.abs(y_true - y_pred)
        mark = tf.cast(tf.less_equal(abs_val, 1.0), tf.float32)

        loss = mark * (abs_val * abs_val * 0.5) + (1 - mark) * (abs_val - 0.5)

        return tf.reduce_sum(loss, axis= -1)

    def log_loss(self, y_true, y_pred):
        y_pred = tf.maximum(y_pred, 1e-8)
        loss = -tf.reduce_sum(tf.math.xlogy(y_true, y_pred), axis= -1)

        return loss

    if DEBUG == 0:
        '''
        [batch, n_priors, 4 + 1 + classes + 4]
        '''
        def calc_loss(self, y_true, y_pred):
            batches = tf.shape(y_true)[0]
            n_boxes = tf.shape(y_true)[1]

            conf_loss = self.log_loss(y_true[:, :, :-8], y_pred[:, :, :-8])
            local_loss = self.L1_smooth_loss(y_true[:, :, -8:-4], y_pred[:, :, -8:-4])

            positives = tf.reduce_max(y_true[:, :, :-9], axis=-1)
            negatives = y_true[:, :, -9]

            n_positive = tf.reduce_sum(positives)

            pos_conf_loss = tf.reduce_sum(conf_loss * positives, axis=-1)
            neg_conf_loss_all = conf_loss * negatives
            n_neg_losses = tf.math.count_nonzero(neg_conf_loss_all, dtype=tf.int32)
            n_negative_keep = tf.minimum(self.neg_pos_ratio * tf.cast(n_positive, tf.int32), n_neg_losses)

            def f1():
                return tf.zeros([batches])

            def f2():
                neg_conf_loss_all_1D = tf.reshape(neg_conf_loss_all, [-1])

                _, indices = tf.nn.top_k(neg_conf_loss_all_1D,
                                         k=n_negative_keep,
                                         sorted=False)

                negatives_keep = tf.scatter_nd(indices= tf.expand_dims(indices, axis=1),
                                               updates=tf.ones_like(indices, dtype=tf.int32),
                                               shape=tf.shape(neg_conf_loss_all_1D))

                negatives_keep = tf.cast(tf.reshape(negatives_keep, [batches, n_boxes]), tf.float32)
                neg_conf_loss = tf.reduce_sum(conf_loss * negatives_keep, axis=-1)
                return neg_conf_loss

            neg_conf_loss = tf.cond(tf.equal(n_neg_losses, 0), f1, f2)
            confidence_loss = pos_conf_loss + neg_conf_loss
            loc_loss = tf.reduce_sum(local_loss * positives, axis=-1)
            total_loss = (confidence_loss + self.alpha * loc_loss) / n_positive

            return total_loss * tf.cast(batches, tf.float32)

    elif DEBUG == 1:
        '''
        [batch, n_priors, 1 + classes + 4]
        '''
        def calc_loss(self, y_true, y_pred):
            batches = tf.shape(y_true)[0]

            conf_loss = self.log_loss(y_true[:, :, :-4], y_pred[:, :, :-4])
            local_loss = self.L1_smooth_loss(y_true[:, :, -4:], y_pred[:, :, -4:])

            positives = tf.reduce_max(y_true[:, :, :-5], axis=-1)
            negatives = y_true[:, :, -5]

            pos_conf_loss = tf.reduce_sum(conf_loss * positives, axis=-1)
            pos_loc_loss = tf.reduce_sum(local_loss * positives, axis=-1)

            n_positives = tf.reduce_sum(positives, axis=-1)
            n_negatives = tf.reduce_sum(negatives, axis=-1)

            fix_positives = tf.where(tf.equal(n_positives, 0.), 30., n_positives)
            n_neg_keep = tf.minimum(self.neg_pos_ratio * fix_positives, n_negatives)
            n_neg_keep = tf.cast(tf.reduce_min(n_neg_keep), tf.int32)

            _, indices = tf.nn.top_k(tf.reduce_max(y_pred[:, :, :-5], axis=-1) * negatives, n_neg_keep)

            neg_idx = tf.expand_dims(tf.range(batches), axis=1)
            neg_idx = tf.expand_dims(tf.tile(neg_idx, (1, n_neg_keep)), axis=2)
            indices = tf.expand_dims(indices, axis=2)
            indices = tf.concat([neg_idx, indices], axis=2)

            neg_conf_loss = tf.reduce_sum(tf.gather_nd(conf_loss, indices), axis=-1)

            total_loss = (pos_conf_loss + neg_conf_loss) / (n_positives + tf.cast(n_neg_keep, tf.float32))
            total_loss += (pos_loc_loss / tf.maximum(n_positives, 1.))

            return total_loss

    elif DEBUG == 2:
        '''
        [batch, n_priors, 1 + classes + 4 + 4]
        '''
        def calc_loss(self, y_true, y_pred):
            conf_loss = self.log_loss(y_true[:, :, :-8], y_pred[:, :, :-8])
            local_loss = self.L1_smooth_loss(y_true[:, :, -8:-4], y_pred[:, :, -8:-4])

            positives = tf.reduce_max(y_true[:, :, 1:-9], axis=-1)
            negatives = y_true[:, :, -9]

            pos_conf_loss = tf.reduce_sum(conf_loss * positives, axis=-1)
            pos_loc_loss = tf.reduce_sum(local_loss * positives, axis=-1)

            n_positives = tf.reduce_sum(positives, axis=-1)
            n_negatives = tf.reduce_sum(negatives, axis=-1)

            fix_positives = tf.where(tf.equal(n_positives, 0.), 100., n_positives)
            n_neg_keep = tf.minimum(self.neg_pos_ratio * fix_positives, n_negatives)
            n_neg_keep = tf.cast(tf.reduce_min(n_neg_keep), tf.int32)

            neg_conf_loss, indices = tf.nn.top_k(conf_loss * negatives, n_neg_keep)
            '''
            neg_idx = tf.expand_dims(tf.range(batches), axis=1)
            neg_idx = tf.expand_dims(tf.tile(neg_idx, (1, n_neg_keep)), axis=2)
            indices = tf.expand_dims(indices, axis=2)
            indices = tf.concat([neg_idx, indices], axis=2)
            
            neg_conf_loss = tf.reduce_sum(tf.gather_nd(conf_loss, indices), axis=-1)
            '''
            total_loss = (pos_conf_loss + tf.reduce_sum(neg_conf_loss, axis=-1)) / (n_positives + tf.cast(n_neg_keep, tf.float32))
            total_loss += (pos_loc_loss / tf.maximum(n_positives, 1.))

            return total_loss

    elif DEBUG == 3:
        '''
        [batch, boxes, classes],  [batch, boxes, classes]
        '''
        def conf_loss(self, y_true, y_pred):
            batches = tf.shape(y_true)[0]

            loss =  self.log_loss(y_true, y_pred)

            positives = tf.reduce_max(y_true[:, :, 1:], axis=-1)
            negatives = y_true[:, :, 0]

            pos_conf_loss = tf.reduce_sum(loss * positives, axis=-1)

            n_positives = tf.reduce_sum(positives, axis=-1)
            n_negatives = tf.reduce_sum(negatives, axis=-1)

            fix_positives = tf.where(tf.equal(n_positives, 0.), 100., n_positives)
            n_neg_keep = tf.minimum(self.neg_pos_ratio * fix_positives, n_negatives)
            n_neg_keep = tf.cast(tf.reduce_min(n_neg_keep), tf.int32)

            _, indices = tf.nn.top_k(tf.reduce_max(y_pred[:, :, 1:], axis=-1) * negatives, n_neg_keep)

            neg_idx = tf.expand_dims(tf.range(batches), axis=1)
            neg_idx = tf.expand_dims(tf.tile(neg_idx, (1, n_neg_keep)), axis=2)
            indices = tf.expand_dims(indices, axis=2)
            indices = tf.concat([neg_idx, indices], axis=2)

            neg_conf_loss = tf.reduce_sum(tf.gather_nd(loss, indices), axis=-1)

            total_loss = (pos_conf_loss + neg_conf_loss) / (n_positives + tf.cast(n_neg_keep, tf.float32))

            return total_loss

        '''
        [batch, boxes, 4(regression) + 1(positive)],   [batch, boxes, 4(regression) + 4(anchors)]
        '''
        def local_loss(self, y_true, y_pred):
            loss = self.L1_smooth_loss(y_true[:, :, :4], y_pred[:, :, :4])

            positives = y_true[:, :, 4]
            pos_loc_loss = tf.reduce_sum(loss * positives, axis=-1)
            n_positives = tf.reduce_sum(positives, axis=-1)

            total_loss = ((self.alpha * pos_loc_loss) / tf.maximum(n_positives, 1.))

            return total_loss







































