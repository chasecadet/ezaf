# coding: utf-8

"""
    Kubeflow Training SDK

    Python SDK for Kubeflow Training  # noqa: E501

    The version of the OpenAPI document: v1.5.0
    Generated by: https://openapi-generator.tech
"""


import pprint
import re  # noqa: F401

import six

from kubeflow.training.configuration import Configuration


class KubeflowOrgV1SchedulingPolicy(object):
    """NOTE: This class is auto generated by OpenAPI Generator.
    Ref: https://openapi-generator.tech

    Do not edit the class manually.
    """

    """
    Attributes:
      openapi_types (dict): The key is attribute name
                            and the value is attribute type.
      attribute_map (dict): The key is attribute name
                            and the value is json key in definition.
    """
    openapi_types = {
        'min_available': 'int',
        'min_resources': 'dict(str, Quantity)',
        'priority_class': 'str',
        'queue': 'str',
        'schedule_timeout_seconds': 'int'
    }

    attribute_map = {
        'min_available': 'minAvailable',
        'min_resources': 'minResources',
        'priority_class': 'priorityClass',
        'queue': 'queue',
        'schedule_timeout_seconds': 'scheduleTimeoutSeconds'
    }

    def __init__(self, min_available=None, min_resources=None, priority_class=None, queue=None, schedule_timeout_seconds=None, local_vars_configuration=None):  # noqa: E501
        """KubeflowOrgV1SchedulingPolicy - a model defined in OpenAPI"""  # noqa: E501
        if local_vars_configuration is None:
            local_vars_configuration = Configuration()
        self.local_vars_configuration = local_vars_configuration

        self._min_available = None
        self._min_resources = None
        self._priority_class = None
        self._queue = None
        self._schedule_timeout_seconds = None
        self.discriminator = None

        if min_available is not None:
            self.min_available = min_available
        if min_resources is not None:
            self.min_resources = min_resources
        if priority_class is not None:
            self.priority_class = priority_class
        if queue is not None:
            self.queue = queue
        if schedule_timeout_seconds is not None:
            self.schedule_timeout_seconds = schedule_timeout_seconds

    @property
    def min_available(self):
        """Gets the min_available of this KubeflowOrgV1SchedulingPolicy.  # noqa: E501


        :return: The min_available of this KubeflowOrgV1SchedulingPolicy.  # noqa: E501
        :rtype: int
        """
        return self._min_available

    @min_available.setter
    def min_available(self, min_available):
        """Sets the min_available of this KubeflowOrgV1SchedulingPolicy.


        :param min_available: The min_available of this KubeflowOrgV1SchedulingPolicy.  # noqa: E501
        :type: int
        """

        self._min_available = min_available

    @property
    def min_resources(self):
        """Gets the min_resources of this KubeflowOrgV1SchedulingPolicy.  # noqa: E501


        :return: The min_resources of this KubeflowOrgV1SchedulingPolicy.  # noqa: E501
        :rtype: dict(str, Quantity)
        """
        return self._min_resources

    @min_resources.setter
    def min_resources(self, min_resources):
        """Sets the min_resources of this KubeflowOrgV1SchedulingPolicy.


        :param min_resources: The min_resources of this KubeflowOrgV1SchedulingPolicy.  # noqa: E501
        :type: dict(str, Quantity)
        """

        self._min_resources = min_resources

    @property
    def priority_class(self):
        """Gets the priority_class of this KubeflowOrgV1SchedulingPolicy.  # noqa: E501


        :return: The priority_class of this KubeflowOrgV1SchedulingPolicy.  # noqa: E501
        :rtype: str
        """
        return self._priority_class

    @priority_class.setter
    def priority_class(self, priority_class):
        """Sets the priority_class of this KubeflowOrgV1SchedulingPolicy.


        :param priority_class: The priority_class of this KubeflowOrgV1SchedulingPolicy.  # noqa: E501
        :type: str
        """

        self._priority_class = priority_class

    @property
    def queue(self):
        """Gets the queue of this KubeflowOrgV1SchedulingPolicy.  # noqa: E501


        :return: The queue of this KubeflowOrgV1SchedulingPolicy.  # noqa: E501
        :rtype: str
        """
        return self._queue

    @queue.setter
    def queue(self, queue):
        """Sets the queue of this KubeflowOrgV1SchedulingPolicy.


        :param queue: The queue of this KubeflowOrgV1SchedulingPolicy.  # noqa: E501
        :type: str
        """

        self._queue = queue

    @property
    def schedule_timeout_seconds(self):
        """Gets the schedule_timeout_seconds of this KubeflowOrgV1SchedulingPolicy.  # noqa: E501


        :return: The schedule_timeout_seconds of this KubeflowOrgV1SchedulingPolicy.  # noqa: E501
        :rtype: int
        """
        return self._schedule_timeout_seconds

    @schedule_timeout_seconds.setter
    def schedule_timeout_seconds(self, schedule_timeout_seconds):
        """Sets the schedule_timeout_seconds of this KubeflowOrgV1SchedulingPolicy.


        :param schedule_timeout_seconds: The schedule_timeout_seconds of this KubeflowOrgV1SchedulingPolicy.  # noqa: E501
        :type: int
        """

        self._schedule_timeout_seconds = schedule_timeout_seconds

    def to_dict(self):
        """Returns the model properties as a dict"""
        result = {}

        for attr, _ in six.iteritems(self.openapi_types):
            value = getattr(self, attr)
            if isinstance(value, list):
                result[attr] = list(map(
                    lambda x: x.to_dict() if hasattr(x, "to_dict") else x,
                    value
                ))
            elif hasattr(value, "to_dict"):
                result[attr] = value.to_dict()
            elif isinstance(value, dict):
                result[attr] = dict(map(
                    lambda item: (item[0], item[1].to_dict())
                    if hasattr(item[1], "to_dict") else item,
                    value.items()
                ))
            else:
                result[attr] = value

        return result

    def to_str(self):
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self):
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other):
        """Returns true if both objects are equal"""
        if not isinstance(other, KubeflowOrgV1SchedulingPolicy):
            return False

        return self.to_dict() == other.to_dict()

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        if not isinstance(other, KubeflowOrgV1SchedulingPolicy):
            return True

        return self.to_dict() != other.to_dict()
