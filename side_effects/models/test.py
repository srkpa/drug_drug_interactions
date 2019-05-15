from torch.nn import AvgPool1d, BatchNorm1d, Conv1d, Dropout, Embedding, MaxPool1d, Sequential, Module
from ivbase.nn.attention import StandardSelfAttention
from ivbase.nn.base import ClonableModule
from ivbase.nn.commons import (
    GlobalAvgPool1d, GlobalMaxPool1d, Transpose, UnitNormLayer, get_activation)


class Cnn1d(ClonableModule):
    r"""
    Extract features from a sequence-like data using Convolutional Neural Network.
    Each time step or position of the sequence must be a discrete value.

    Arguments
    ----------
        vocab_size: int
            Size of the vocabulary, i.e the maximum number of discrete elements possible at each time step.
            Since padding will be used for small sequences, we expect the vocab size to be 1 + size of the alphabet.
            We also expect that 0 won't be use to represent any element of the vocabulary expect the padding.
        embedding_size: int
            The size of each embedding vector
        cnn_sizes: int list
            A list that specifies the size of each convolution layer.
            The size of the list implicitly defines the number of layers of the network
        kernel_size: int or list(int)
            he size of the kernel, i.e the number of time steps include in one convolution operation.
            An integer means the same value will be used for each conv layer. A list allows to specify different sizes for different layers.
            The length of the list should match the length of cnn_sizes.
        pooling_len: int or int list, optional
            The number of time steps aggregated together by the pooling operation.
            An integer means the same pooling length is used for all layers.
            A list allows to specify different length for different layers. The length of the list should match the length of cnn_sizes
            (Default value = 1)
        pooling: str, optional
            One of {'avg', 'max'} (for AveragePooling and MaxPooling).
            It indicates the type of pooling operator to use after convolution.
            (Default value = 'avg')
        dilatation_rate: int or int list, optional
            The dilation factor tells how large are the gaps between elements in
            a feature map on which we apply a convolution filter.  If a integer is provided, the same value is used for all
            convolution layer. If dilation = 1 (no gaps),  every 1st element next to one position is included in the conv op.
            If dilation = 2, we take every 2nd (gaps of size 1), and so on. See https://arxiv.org/pdf/1511.07122.pdf for more info.
            (Default value = 1)
        activation: str or callable, optional
            The activation function. activation layer {'ReLU', 'Sigmoid', 'Tanh', 'ELU', 'SELU', 'GLU', 'LeakyReLU', 'Softplus'}
            The name of the activation function
        normalize_features: bool, optional
            Whether the extracted features should be unit normalized. It is preferable to normalize later in your model.
            (Default value = False)
        b_norm: bool, optional
            Whether to use Batch Normalization after each convolution layer.
            (Default value = False)
        use_self_attention: bool, optional
            Whether to use a self attention mechanism on the last conv layer before the pooling
            (Default value = False)
        dropout: float, optional
            Dropout probability to regularize the network. No dropout by default.
            (Default value = .0)

    Attributes
    ----------
        extractor: torch.nn.Module
            The underlying feature extractor of the model.
    """

    def __init__(self, vocab_size, embedding_size, cnn_sizes, kernel_size, pooling_len=1, pooling='avg',
                 dilatation_rate=1, activation='ReLU', normalize_features=True, b_norm=False,
                 use_self_attention=False, dropout=0.0):
        super(Cnn1d, self).__init__()
        self.__params = locals()
        activation_cls = get_activation(activation)
        if not isinstance(pooling_len, (list, int)):
            raise TypeError("pooling_len should be of type int or int list")
        if pooling not in ['avg', 'max']:
            raise ValueError("the pooling type must be either 'max' or 'avg'")
        if len(cnn_sizes) <= 0:
            raise ValueError(
                "There should be at least on convolution layer (cnn_size should be positive.)")

        if isinstance(pooling_len, int):
            pooling_len = [pooling_len] * len(cnn_sizes)
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * len(cnn_sizes)
        if pooling == 'avg':
            pool1d = AvgPool1d
            gpool = GlobalAvgPool1d(dim=1)
        else:
            pool1d = MaxPool1d
            gpool = GlobalMaxPool1d(dim=1)

        # network construction

        layers = []
        in_channels = [2] + cnn_sizes[:-1]
        for i, (in_channel, out_channel, ksize, l_pool) in \
                enumerate(zip(in_channels, cnn_sizes, kernel_size, pooling_len)):
            pad = ((dilatation_rate**i) * (ksize - 1) + 1) // 2
            layers.append(Conv1d(in_channel, out_channel, padding=pad,
                                 kernel_size=ksize, dilation=dilatation_rate**i))
            if b_norm:
                layers.append(BatchNorm1d(out_channel))
            layers.append(activation_cls)
            layers.append(Dropout(dropout))
            if l_pool > 1:
                layers.append(pool1d(l_pool))

        if use_self_attention:
            gpool = StandardSelfAttention(
                cnn_sizes[-1], cnn_sizes[-1], pooling)

        layers.append(Transpose(1, 2))
        layers.append(gpool)
        if normalize_features:
            layers.append(UnitNormLayer())

        self.__output_dim = cnn_sizes[-1]
        self.extractor = Sequential(*layers)

    @property
    def output_dim(self):
        r"""
        Get the dimension of the feature space in which the sequences are projected

        Returns
        -------
        output_dim (int): Dimension of the output feature space

        """
        return self.__output_dim

    def forward(self, x):
        r"""
        Forward-pass method

        Arguments
        ----------
            x (torch.LongTensor of size N*L): Batch of N sequences of size L each.
                L is actually the length of the longest of the sequence in the bacth and we expected the
                rest of the sequences to be padded with zeros up to that length.
                Each entry of the tensor is supposed to be an integer representing an element of the vocabulary.
                0 is reserved as the padding marker.

        Returns
        -------
            phi_x: torch.FloatTensor of size N*D
                Batch of feature vectors. D is the dimension of the feature space.
                D is given by output_dim.
        """
        return self.extractor(x)
