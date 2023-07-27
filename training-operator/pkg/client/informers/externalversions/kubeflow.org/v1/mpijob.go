// Copyright 2023 The Kubeflow Authors
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

// Code generated by informer-gen. DO NOT EDIT.

package v1

import (
	"context"
	time "time"

	kubefloworgv1 "github.com/kubeflow/training-operator/pkg/apis/kubeflow.org/v1"
	versioned "github.com/kubeflow/training-operator/pkg/client/clientset/versioned"
	internalinterfaces "github.com/kubeflow/training-operator/pkg/client/informers/externalversions/internalinterfaces"
	v1 "github.com/kubeflow/training-operator/pkg/client/listers/kubeflow.org/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	runtime "k8s.io/apimachinery/pkg/runtime"
	watch "k8s.io/apimachinery/pkg/watch"
	cache "k8s.io/client-go/tools/cache"
)

// MPIJobInformer provides access to a shared informer and lister for
// MPIJobs.
type MPIJobInformer interface {
	Informer() cache.SharedIndexInformer
	Lister() v1.MPIJobLister
}

type mPIJobInformer struct {
	factory          internalinterfaces.SharedInformerFactory
	tweakListOptions internalinterfaces.TweakListOptionsFunc
	namespace        string
}

// NewMPIJobInformer constructs a new informer for MPIJob type.
// Always prefer using an informer factory to get a shared informer instead of getting an independent
// one. This reduces memory footprint and number of connections to the server.
func NewMPIJobInformer(client versioned.Interface, namespace string, resyncPeriod time.Duration, indexers cache.Indexers) cache.SharedIndexInformer {
	return NewFilteredMPIJobInformer(client, namespace, resyncPeriod, indexers, nil)
}

// NewFilteredMPIJobInformer constructs a new informer for MPIJob type.
// Always prefer using an informer factory to get a shared informer instead of getting an independent
// one. This reduces memory footprint and number of connections to the server.
func NewFilteredMPIJobInformer(client versioned.Interface, namespace string, resyncPeriod time.Duration, indexers cache.Indexers, tweakListOptions internalinterfaces.TweakListOptionsFunc) cache.SharedIndexInformer {
	return cache.NewSharedIndexInformer(
		&cache.ListWatch{
			ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
				if tweakListOptions != nil {
					tweakListOptions(&options)
				}
				return client.KubeflowV1().MPIJobs(namespace).List(context.TODO(), options)
			},
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				if tweakListOptions != nil {
					tweakListOptions(&options)
				}
				return client.KubeflowV1().MPIJobs(namespace).Watch(context.TODO(), options)
			},
		},
		&kubefloworgv1.MPIJob{},
		resyncPeriod,
		indexers,
	)
}

func (f *mPIJobInformer) defaultInformer(client versioned.Interface, resyncPeriod time.Duration) cache.SharedIndexInformer {
	return NewFilteredMPIJobInformer(client, f.namespace, resyncPeriod, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc}, f.tweakListOptions)
}

func (f *mPIJobInformer) Informer() cache.SharedIndexInformer {
	return f.factory.InformerFor(&kubefloworgv1.MPIJob{}, f.defaultInformer)
}

func (f *mPIJobInformer) Lister() v1.MPIJobLister {
	return v1.NewMPIJobLister(f.Informer().GetIndexer())
}
