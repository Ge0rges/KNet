import np
import torch

def equations_loader(batch_size=256, num_workers=0, pin_memory=False):
    eqs = [
        lambda inputs: 1.0 if np.sum(inputs) < (len(inputs) * 0.25) else 0.0,
        lambda inputs: 1.0 if (len(inputs) * 0.5) > np.sum(inputs) > (len(inputs) * 0.25) else 0.0,
        lambda inputs: 1.0 if (len(inputs) * 0.75) > np.sum(inputs) > (len(inputs) * 0.5) else 0.0,
        lambda inputs: 1.0 if np.sum(inputs) > (len(inputs) * 0.75) else 0.0
    ]

    inputs = np.random.rand(10000, 10).astype('f')
    targets = np.asarray([[eq(i) for eq in eqs] for i in inputs]).astype('f')

    dataset = torch.utils.data.TensorDataset(torch.from_numpy(inputs), torch.from_numpy(targets))

    sampler = RandomSampler(dataset)
    loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    return loader
