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
from kubeflow.training.models.kubeflow_org_v1_run_policy import KubeflowOrgV1RunPolicy  # noqa: E501
from kubeflow.training.rest import ApiException

class TestKubeflowOrgV1RunPolicy(unittest.TestCase):
    """KubeflowOrgV1RunPolicy unit test stubs"""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def make_instance(self, include_optional):
        """Test KubeflowOrgV1RunPolicy
            include_option is a boolean, when False only required
            params are included, when True both required and
            optional params are included """
        # model = kubeflow.training.models.kubeflow_org_v1_run_policy.KubeflowOrgV1RunPolicy()  # noqa: E501
        if include_optional :
            return KubeflowOrgV1RunPolicy(
                active_deadline_seconds = 56, 
                backoff_limit = 56, 
                clean_pod_policy = '0', 
                scheduling_policy = kubeflow_org_v1_scheduling_policy.KubeflowOrgV1SchedulingPolicy(
                    min_available = 56, 
                    min_resources = {
                        'key' : None
                        }, 
                    priority_class = '0', 
                    queue = '0', 
                    schedule_timeout_seconds = 56, ), 
                ttl_seconds_after_finished = 56
            )
        else :
            return KubeflowOrgV1RunPolicy(
        )

    def testKubeflowOrgV1RunPolicy(self):
        """Test KubeflowOrgV1RunPolicy"""
        inst_req_only = self.make_instance(include_optional=False)
        inst_req_and_optional = self.make_instance(include_optional=True)


if __name__ == '__main__':
    unittest.main()
