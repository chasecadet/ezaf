// Copyright 2021 The Kubeflow Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package pytorch

import (
	"context"
	"fmt"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/pointer"

	kubeflowv1 "github.com/kubeflow/training-operator/pkg/apis/kubeflow.org/v1"
	"github.com/kubeflow/training-operator/pkg/util/testutil"
)

var _ = Describe("PyTorchJob controller", func() {
	// Define utility constants for object names.
	const (
		expectedPort = int32(8080)
	)

	Context("When creating the PyTorchJob", func() {
		It("Should get the corresponding resources successfully", func() {
			const (
				namespace = "default"
				name      = "test-job"
			)
			By("By creating a new PyTorchJob")
			ctx := context.Background()
			job := newPyTorchJobForTest(name, namespace)
			job.Spec.PyTorchReplicaSpecs = map[kubeflowv1.ReplicaType]*kubeflowv1.ReplicaSpec{
				kubeflowv1.PyTorchJobReplicaTypeMaster: {
					Replicas: pointer.Int32(1),
					Template: corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{
								{
									Image: "test-image",
									Name:  kubeflowv1.PytorchJobDefaultContainerName,
									Ports: []corev1.ContainerPort{
										{
											Name:          kubeflowv1.PytorchJobDefaultPortName,
											ContainerPort: expectedPort,
											Protocol:      corev1.ProtocolTCP,
										},
									},
								},
							},
						},
					},
				},
				kubeflowv1.PyTorchJobReplicaTypeWorker: {
					Replicas: pointer.Int32(2),
					Template: corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{
								{
									Image: "test-image",
									Name:  kubeflowv1.PytorchJobDefaultContainerName,
									Ports: []corev1.ContainerPort{
										{
											Name:          kubeflowv1.PytorchJobDefaultPortName,
											ContainerPort: expectedPort,
											Protocol:      corev1.ProtocolTCP,
										},
									},
								},
							},
						},
					},
				},
			}
			job.Spec.NprocPerNode = nil

			Expect(testK8sClient.Create(ctx, job)).Should(Succeed())

			key := types.NamespacedName{Name: name, Namespace: namespace}
			created := &kubeflowv1.PyTorchJob{}

			// We'll need to retry getting this newly created PyTorchJob, given that creation may not immediately happen.
			Eventually(func() bool {
				err := testK8sClient.Get(ctx, key, created)
				return err == nil
			}, testutil.Timeout, testutil.Interval).Should(BeTrue())

			masterKey := types.NamespacedName{Name: fmt.Sprintf("%s-master-0", name), Namespace: namespace}
			masterPod := &corev1.Pod{}
			Eventually(func() bool {
				err := testK8sClient.Get(ctx, masterKey, masterPod)
				return err == nil
			}, testutil.Timeout, testutil.Interval).Should(BeTrue())

			masterSvc := &corev1.Service{}
			Eventually(func() bool {
				err := testK8sClient.Get(ctx, masterKey, masterSvc)
				return err == nil
			}, testutil.Timeout, testutil.Interval).Should(BeTrue())

			// Check the pod port.
			Expect(masterPod.Spec.Containers[0].Ports).To(ContainElement(corev1.ContainerPort{
				Name:          kubeflowv1.PytorchJobDefaultPortName,
				ContainerPort: expectedPort,
				Protocol:      corev1.ProtocolTCP}))
			// Check env variable
			Expect(masterPod.Spec.Containers[0].Env).To(ContainElements(corev1.EnvVar{
				Name:  EnvMasterPort,
				Value: fmt.Sprintf("%d", masterSvc.Spec.Ports[0].Port),
			}, corev1.EnvVar{
				Name:  EnvMasterAddr,
				Value: masterSvc.Name,
			}, corev1.EnvVar{
				Name:  EnvNprocPerNode,
				Value: kubeflowv1.DefaultNprocPerNode,
			}))
			// Check service port.
			Expect(masterSvc.Spec.Ports[0].Port).To(Equal(expectedPort))
			// Check owner reference.
			trueVal := true
			Expect(masterPod.OwnerReferences).To(ContainElement(metav1.OwnerReference{
				APIVersion:         kubeflowv1.SchemeGroupVersion.String(),
				Kind:               kubeflowv1.PytorchJobKind,
				Name:               name,
				UID:                created.UID,
				Controller:         &trueVal,
				BlockOwnerDeletion: &trueVal,
			}))
			Expect(masterSvc.OwnerReferences).To(ContainElement(metav1.OwnerReference{
				APIVersion:         kubeflowv1.SchemeGroupVersion.String(),
				Kind:               kubeflowv1.PytorchJobKind,
				Name:               name,
				UID:                created.UID,
				Controller:         &trueVal,
				BlockOwnerDeletion: &trueVal,
			}))

			// Test job status.
			masterPod.Status.Phase = corev1.PodSucceeded
			masterPod.ResourceVersion = ""
			Expect(testK8sClient.Status().Update(ctx, masterPod)).Should(Succeed())
			Eventually(func() bool {
				err := testK8sClient.Get(ctx, key, created)
				if err != nil {
					return false
				}
				return created.Status.ReplicaStatuses != nil && created.Status.
					ReplicaStatuses[kubeflowv1.PyTorchJobReplicaTypeMaster].Succeeded == 1
			}, testutil.Timeout, testutil.Interval).Should(BeTrue())
			// Check if the job is succeeded.
			cond := getCondition(created.Status, kubeflowv1.JobSucceeded)
			Expect(cond.Status).To(Equal(corev1.ConditionTrue))
			By("Deleting the PyTorchJob")
			Expect(testK8sClient.Delete(ctx, job)).Should(Succeed())
		})
	})

	Context("When creating the elastic PyTorchJob", func() {
		// TODO(gaocegege): Test with more than 1 worker.
		It("Should get the corresponding resources successfully", func() {
			// Define the expected elastic policy.
			var (
				backendC10D = kubeflowv1.BackendC10D
				minReplicas = pointer.Int32(1)
				maxReplicas = pointer.Int32(3)
				maxRestarts = pointer.Int32(3)
				namespace   = "default"
				name        = "easltic-job"
			)

			By("By creating a new PyTorchJob")
			ctx := context.Background()
			job := newPyTorchJobForTest(name, namespace)
			job.Spec.ElasticPolicy = &kubeflowv1.ElasticPolicy{
				RDZVBackend: &backendC10D,
				MaxReplicas: maxReplicas,
				MinReplicas: minReplicas,
				MaxRestarts: maxRestarts,
			}
			job.Spec.PyTorchReplicaSpecs = map[kubeflowv1.ReplicaType]*kubeflowv1.ReplicaSpec{
				kubeflowv1.PyTorchJobReplicaTypeWorker: {
					Replicas: pointer.Int32(1),
					Template: corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{
								{
									Image: "test-image",
									Name:  kubeflowv1.PytorchJobDefaultContainerName,
									Ports: []corev1.ContainerPort{
										{
											Name:          kubeflowv1.PytorchJobDefaultPortName,
											ContainerPort: expectedPort,
											Protocol:      corev1.ProtocolTCP,
										},
									},
								},
							},
						},
					},
				},
			}

			Expect(testK8sClient.Create(ctx, job)).Should(Succeed())

			key := types.NamespacedName{Name: name, Namespace: namespace}
			created := &kubeflowv1.PyTorchJob{}

			// We'll need to retry getting this newly created PyTorchJob, given that creation may not immediately happen.
			Eventually(func() bool {
				err := testK8sClient.Get(ctx, key, created)
				return err == nil
			}, testutil.Timeout, testutil.Interval).Should(BeTrue())

			workerKey := types.NamespacedName{Name: fmt.Sprintf("%s-worker-0", name), Namespace: namespace}
			pod := &corev1.Pod{}
			Eventually(func() bool {
				err := testK8sClient.Get(ctx, workerKey, pod)
				return err == nil
			}, testutil.Timeout, testutil.Interval).Should(BeTrue())

			svc := &corev1.Service{}
			Eventually(func() bool {
				err := testK8sClient.Get(ctx, workerKey, svc)
				return err == nil
			}, testutil.Timeout, testutil.Interval).Should(BeTrue())

			// Check pod port.
			Expect(pod.Spec.Containers[0].Ports).To(ContainElement(corev1.ContainerPort{
				Name:          kubeflowv1.PytorchJobDefaultPortName,
				ContainerPort: expectedPort,
				Protocol:      corev1.ProtocolTCP}))
			// Check environment variables.
			Expect(pod.Spec.Containers[0].Env).To(ContainElements(corev1.EnvVar{
				Name:  EnvRDZVBackend,
				Value: string(backendC10D),
			}, corev1.EnvVar{
				Name:  EnvNnodes,
				Value: fmt.Sprintf("%d:%d", *minReplicas, *maxReplicas),
			}, corev1.EnvVar{
				Name:  EnvRDZVEndpoint,
				Value: fmt.Sprintf("%s:%d", svc.Name, expectedPort),
			}, corev1.EnvVar{
				Name:  EnvMaxRestarts,
				Value: fmt.Sprintf("%d", *maxRestarts),
			}))
			Expect(svc.Spec.Ports[0].Port).To(Equal(expectedPort))
			// Check owner references.
			trueVal := true
			Expect(pod.OwnerReferences).To(ContainElement(metav1.OwnerReference{
				APIVersion:         kubeflowv1.SchemeGroupVersion.String(),
				Kind:               kubeflowv1.PytorchJobKind,
				Name:               name,
				UID:                created.UID,
				Controller:         &trueVal,
				BlockOwnerDeletion: &trueVal,
			}))
			Expect(svc.OwnerReferences).To(ContainElement(metav1.OwnerReference{
				APIVersion:         kubeflowv1.SchemeGroupVersion.String(),
				Kind:               kubeflowv1.PytorchJobKind,
				Name:               name,
				UID:                created.UID,
				Controller:         &trueVal,
				BlockOwnerDeletion: &trueVal,
			}))

			// Test job status.
			pod.Status.Phase = corev1.PodSucceeded
			pod.ResourceVersion = ""
			Expect(testK8sClient.Status().Update(ctx, pod)).Should(Succeed())
			Eventually(func() bool {
				err := testK8sClient.Get(ctx, key, created)
				if err != nil {
					return false
				}
				return created.Status.ReplicaStatuses != nil && created.Status.
					ReplicaStatuses[kubeflowv1.PyTorchJobReplicaTypeWorker].Succeeded == 1
			}, testutil.Timeout, testutil.Interval).Should(BeTrue())
			// Check if the job is succeeded.
			cond := getCondition(created.Status, kubeflowv1.JobSucceeded)
			Expect(cond.Status).To(Equal(corev1.ConditionTrue))
			By("Deleting the PyTorchJob")
			Expect(testK8sClient.Delete(ctx, job)).Should(Succeed())
		})
	})
})

func newPyTorchJobForTest(name, namespace string) *kubeflowv1.PyTorchJob {
	return &kubeflowv1.PyTorchJob{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
	}
}

// getCondition returns the condition with the provided type.
func getCondition(status kubeflowv1.JobStatus, condType kubeflowv1.JobConditionType) *kubeflowv1.JobCondition {
	for _, condition := range status.Conditions {
		if condition.Type == condType {
			return &condition
		}
	}
	return nil
}
