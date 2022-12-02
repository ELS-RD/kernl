import dataclasses
import torch


@dataclasses.dataclass
class RegisteredStorage:
    storage: torch.Storage
    size: int
    ptr: int

    @property
    def end_ptr(self) -> int:
        return self.ptr + self.size

    @property
    def access_tensor(self) -> torch.Tensor:
        return torch.tensor(self.storage, dtype=self.storage.dtype, device=self.storage.device)

    def ensure_immutable(self):
        assert self.storage.data_ptr() == self.ptr and self.storage.size() == self.size


class MemoryMap:
    storages: [RegisteredStorage]

    def __init__(self):
        self.storages = []

    def _get_registered_storage(self, pointer: torch.Tensor):
        max_pointer = torch.max(pointer).item()
        min_pointer = torch.min(pointer).item()
        registered_storage = next(
            filter(lambda registered: min_pointer >= registered.ptr and max_pointer < registered.end_ptr,
                   self.storages), None)
        if registered_storage is None:
            raise Exception("Storage not found or pointers spanning multiple tensors")
        registered_storage.ensure_immutable()
        return registered_storage

    def add_tensor(self, t: torch.Tensor):
        storage = t.storage()
        self.storages.append(RegisteredStorage(storage, storage.size(), storage.data_ptr()))
        return t.data_ptr()

    def load(self,
             pointer: torch.Tensor,
             mask: torch.Tensor = None,
             other=0.0,
             ):
        assert 0 < pointer.dim() < 3
        assert pointer.dtype == torch.int64

        if mask is None:
            mask = torch.ones_like(pointer).bool()
        assert 0 < mask.dim() < 3
        assert mask.dtype == torch.bool

        registered_storage = self._get_registered_storage(pointer)
        access_tensor = registered_storage.access_tensor

        index_tensor = (pointer - registered_storage.ptr)

        block = torch.full_like(pointer, fill_value=other, dtype=access_tensor.dtype, device="cuda")
        block[mask] = access_tensor[index_tensor][mask]
        return block

    def store(self,
              pointer: torch.Tensor,
              value: torch.Tensor,
              mask=None):

        assert 0 < pointer.dim() < 3
        assert pointer.dtype == torch.int64

        if mask is None:
            mask = torch.ones_like(pointer).bool()
        assert 0 < mask.dim() < 3
        assert mask.dtype == torch.bool

        registered_storage = self._get_registered_storage(pointer)
        access_tensor = registered_storage.access_tensor

        index_tensor = (pointer - registered_storage.ptr)
        access_tensor[index_tensor[mask]] = value[mask]