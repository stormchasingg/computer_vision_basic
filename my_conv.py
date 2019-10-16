import numpy as np


class myConv(object):
    
    def __init__(self, input_data, weights_data, stride, padding='SAME'):
        self.input = np.asarray(input_data, np.float32)
        self.weights = np.asarray(weights_data, np.float32)
        self.stride = stride
        self.padding = padding
        
    def compute_conv(self, fm, kernel):
        [h,w] = fm.shape
        [k,_] = kernel.shape
 
        if self.padding == 'SAME':
            pad_h = (self.stride * (h-1) + k - h) // 2
            pad_w = (self.stride * (w-1) + k - w) // 2
            rs_h = h
            rs_w = w
        elif self.padding == 'VALID':
            pad_h = 0
            pad_w = 0
            rs_h = (h-k) / self.stride + 1
            rs_w = (w-k) / self.stride + 1
        elif self.padding == 'FULL':
            pad_h = k - 1
            pad_w = k - 1
            rs_h = (h+k-2) / self.stride + 1
            rs_w = (w+k-2) / self.stride + 1
        else:
            pad_h = 0
            pad_w = 0
            rs_h = (h-k) / self.stride + 1
            rs_w = (w-k) / self.stride + 1
        
        padding_fm = np.zeros([h + 2*pad_h, w + 2*pad_w], np.float32)
        padding_fm[pad_h:pad_h+h, pad_w:pad_w+w] = fm
        rs = np.zeros([rs_h,rs_w],np.float32)
 
        for i in range(rs_h):
            for j in range(rs_w):
                roi = padding_fm[i*self.stride:(i*self.stride+k), j*self.stride:(j*self.stride+k)]
                rs[i][j] = np.sum(roi * kernel)
        return rs
 
    def my_conv2d(self):
        """
        self.input:c*h*w
        self.weights:c*h*w
        :return:
        """
        [c,h,w] = self.input.shape
        [kc,k,_] = self.weights.shape
        assert c == kc
        outputs = []
        for i in range(c):
            f_map = self.input[i]
            kernel = self.weights[i]
            rs = self.compute_conv(f_map, kernel)
            # print(rs)
            if i == 0:
                outputs = rs
            else:
                outputs += rs
        return outputs


if __name__ == '__main__':
    # c*h*w shape=[c,h,w]
    input_data = [
        [[1, 0, 1, 2, 1],
         [0, 2, 1, 0, 1],
         [1, 1, 0, 2, 0],
         [2, 2, 1, 1, 0],
         [2, 0, 1, 2, 0]],
        [[2, 0, 2, 1, 1],
         [0, 1, 0, 0, 2],
         [1, 0, 0, 2, 1],
         [1, 1, 2, 1, 0],
         [1, 0, 1, 1, 1]],
    ]
    # in_c*k*k
    weights_data = [
        [[1, 0, 1],
         [-1, 1, 0],
         [0, -1, 0]],
        [[-1, 0, 1],
         [0, 0, 1],
         [1, 1, 1]]
    ]

    conv = myConv(input_data, weights_data, 1, 'SAME')
    print(conv.my_conv2d())
    
