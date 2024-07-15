
"""
recbole_cdr.model.crossdomain_recommender
##################################
"""
import torch
from recbole.model.abstract_recommender import AbstractRecommender
from recbole_cdr.utils import ModelType


class CrossDomainRecommender(AbstractRecommender):
    """This is a abstract cross-domain recommender. All the cross-domain model should implement this class.
    The base cross-domain recommender class provide the basic dataset and parameters information.
    """

    type = ModelType.CROSSDOMAIN

    def __init__(self, config, dataset):
        super(CrossDomainRecommender, self).__init__()

        # load source dataset info
        self.SOURCE_USER_ID = dataset.source_domain_dataset.uid_field
        self.SOURCE_ITEM_ID = dataset.source_domain_dataset.iid_field
        self.SOURCE_NEG_ITEM_ID = config['source_domain']['NEG_PREFIX'] + self.SOURCE_ITEM_ID
        self.source_num_users = dataset.source_domain_dataset.num(self.SOURCE_USER_ID)
        self.source_num_items = dataset.source_domain_dataset.num(self.SOURCE_ITEM_ID)

        # load target dataset info
        self.TARGET_USER_ID = dataset.target_domain_dataset.uid_field
        self.TARGET_ITEM_ID = dataset.target_domain_dataset.iid_field
        self.TARGET_NEG_ITEM_ID = config['target_domain']['NEG_PREFIX'] + self.TARGET_ITEM_ID
        self.target_num_users = dataset.target_domain_dataset.num(self.TARGET_USER_ID)
        self.target_num_items = dataset.target_domain_dataset.num(self.TARGET_ITEM_ID)

        # load both dataset info
        self.total_num_users = dataset.num_total_user
        self.total_num_items = dataset.num_total_item

        self.overlapped_num_users = dataset.num_overlap_user
        self.overlapped_num_items = dataset.num_overlap_item

        self.OVERLAP_ID = dataset.overlap_id_field

        # load parameters info
        self.device = config['device']

    def set_phase(self, phase):
        pass

    def epoch_start(self):
        pass

    def set_full_sort_func(self, scheme='Source'):
        if scheme == 'Source':
            self.full_sort_predict = self.full_sort_predict_source
        elif scheme == 'Target':
            self.full_sort_predict = self.full_sort_predict_target

    def set_predict_func(self, scheme='Source'):
        if scheme == 'Source':
            self.predict = self.predict_source
        elif scheme == 'Target':
            self.predict = self.predict_target

    def full_sort_predict_source(self, interaction):
        raise NotImplementedError

    def full_sort_predict_target(self, interaction):
        raise NotImplementedError

    def predict_source(self, interaction):
        raise NotImplementedError

    def predict_target(self, interaction):
        raise NotImplementedError

class AutoEncoderMixin(object):
    """This is a common part of auto-encoders. All the auto-encoder models should inherit this class,
    including CDAE, MacridVAE, MultiDAE, MultiVAE, RaCT and RecVAE.
    The base AutoEncoderMixin class provides basic dataset information and rating matrix function.
    """

    def convert_sparse_matrix_to_rating_matrix(self, spmatrix):
        rating = spmatrix.toarray()
        self.rating_matrix = torch.tensor(rating)

    def build_histroy_items(self, dataset):
        self.history_item_id, self.history_item_value, _ = dataset.history_item_matrix()

    def get_rating_matrix(self, user):
        r"""Get a batch of user's feature with the user's id and history interaction matrix.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The user's feature of a batch of user, shape: [batch_size, n_items]
        """
        # Following lines construct tensor of shape [B,n_items] using the tensor of shape [B,H]
        col_indices = self.history_item_id[user.cpu()].flatten()
        row_indices = torch.arange(user.shape[0]).repeat_interleave(
            self.history_item_id.shape[1], dim=0
        )
        rating_matrix = torch.zeros(1).repeat(user.shape[0], self.total_num_items)
        rating_matrix.index_put_(
            (row_indices, col_indices), self.history_item_value[user.cpu()].flatten()
        )
        rating_matrix = rating_matrix.to(self.device)
        return rating_matrix