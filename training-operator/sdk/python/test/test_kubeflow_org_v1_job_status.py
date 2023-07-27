# coding: utf-8

"""
    Kubeflow Training SDK

    Python SDK for Kubeflow Training  # noqa: E501

    The version of the OpenAPI document: v1.5.0
    Generated by: https://openapi-generator.tech
"""


from __future__ import absolute_import

import unittest
import datetime

from kubeflow.training.models import *
from kubeflow.training.models.kubeflow_org_v1_job_status import KubeflowOrgV1JobStatus  # noqa: E501
from kubeflow.training.rest import ApiException

class TestKubeflowOrgV1JobStatus(unittest.TestCase):
    """KubeflowOrgV1JobStatus unit test stubs"""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def make_instance(self, include_optional):
        """Test KubeflowOrgV1JobStatus
            include_option is a boolean, when False only required
            params are included, when True both required and
            optional params are included """
        # model = kubeflow.training.models.kubeflow_org_v1_job_status.KubeflowOrgV1JobStatus()  # noqa: E501
        if include_optional :
            return KubeflowOrgV1JobStatus(
                completion_time = None, 
                conditions = [
                    kubeflow_org_v1_job_condition.KubeflowOrgV1JobCondition(
                        last_transition_time = None, 
                        last_update_time = None, 
                        message = '0', 
                        reason = '0', 
                        status = '0', 
                        type = '0', )
                    ], 
                last_reconcile_time = None, 
                replica_statuses = {
                    'key' : kubeflow_org_v1_replica_status.KubeflowOrgV1ReplicaStatus(
                        active = 56, 
                        failed = 56, 
                        label_selector = None, 
                        selector = '0', 
                        succeeded = 56, )
                    }, 
                start_time = None
            )
        else :
            return KubeflowOrgV1JobStatus(
        )

    def testKubeflowOrgV1JobStatus(self):
        """Test KubeflowOrgV1JobStatus"""
        inst_req_only = self.make_instance(include_optional=False)
        inst_req_and_optional = self.make_instance(include_optional=True)


if __name__ == '__main__':
    unittest.main()
