"""
Note: This is a basic implementation of Wagner's k-list algorithm for the strict generalized birthday problem
where 'strict' means the the messages must come from k distinct lists.
"""

import hashlib
from math import log2
import os
from collections import deque
from tqdm import trange
import time
from collections import defaultdict


def blake2b(data : bytes, digest_byte_size : int = 64) -> bytes:
    return hashlib.blake2b(data, digest_size = digest_byte_size).digest()

class KListSolver:
    def __init__(self, n, k, nonce: bytes = None, hashfunc = None):
        """ Init a k-list solver for generalized birthday problem with parameters n, k. 
        The underlying hash function is selected by `hashfunc` function.
        
        Strict generalized birthday problem: Given a hash function h with output length n, 
        find k messages from pre-defined k distinct lists such that XOR of their hash values is 0.
        
        Args:
            n (int): the output bit length of hash function
            k (int): k messages to find, must be a power of 2
            nonce (bytes): the nonce used to generate the hash values
            hashfunc (function): the hash function to use
        """
        assert n % 8 == 0, "n should be a multiple of 8"
        nonce = os.urandom(16) if nonce is None else nonce
        assert len(nonce) == 16, "Nonce should be 16 bytes"
        if hashfunc is None:
            hashfunc = lambda x: blake2b(x, n // 8)
        assert len(hashfunc(b'')) == n // 8, "Hash function output length should be n/8 bytes"
        self.n = n
        self.k = k
        self.lgk = int(log2(k))
        assert 2 ** self.lgk == k, "k should be a power of 2"
        self.nonce = nonce
        self.hashfunc = hashfunc
        self.hash_size = n // 8
        self.mask_bit = (self.n + self.lgk) // (1 + self.lgk) # ceil(n / (lgk + 1))
        self.mask = (1 << self.mask_bit) - 1
        
    @staticmethod
    def new(n, k, nonce = None, hashfunc = None):
        return KListSolver(n, k, nonce, hashfunc)
    
    def merge_join(self, L1: list, L2: list) -> list:
        """ Merge two lists of messages at a certain height in the binary tree. Using sorted list to perform binary search.
        Time complexity: O(n log n)
        Memory complexity: O(n)
        This function has been abandoned and not completed.
        ** For Wagner's k-list algorithm, the hash_join is more efficient and convenient. **
        
        Args:
            L1 (List): the first list of messages
            L2 (List): the second list of messages
            height (int): the height of the binary tree
        
        Returns:
            List[bytes]: the merged list of messages
        """
        L1.sort(key= lambda x: x[0] & self.mask)
        length1 = len(L1)
        
        merged_list = []
        # binary search 
        for x, idx2s in L2:
            xm = x & self.mask
            lb, ub = 0, length1
            while lb < ub:
                m = (lb + ub) // 2
                if L1[m][0] & self.mask < xm:
                    lb = m + 1
                else:
                    ub = m
            if lb < len(L1) and (L1[lb][0]^x) & self.mask == 0:
                merge_x = (L1[lb][0] ^ x) >> self.mask_bit
                merge_idxs = L1[lb][1] + idx2s
                merged_list.append((merge_x, merge_idxs))
        return merged_list
        
    def hash_join(self, L1: list, L2: list, mask_bit: int) -> list:
        """ Perform hash join on two lists of messages.
        Time complexity: O(n)
        Memory complexity: O(n)
        
        Args:
            L1 (List): the first list of messages
            L2 (List): the second list of messages
            mask_bit (int): the number of bits to collide
        
        Returns:
            List[bytes]: the merged list of messages
        """
        hash_table = defaultdict(list)
        mask = (1 << mask_bit) - 1
        
        for x1, idx1 in L1:
            hash_table[x1 & mask].append((x1, idx1))
        merged_list = []
        for x2, idx2 in L2:
            if x2 & mask in hash_table:
                colls = hash_table[x2 & mask]
                for x1, idx1 in colls:
                    merge_x = (x1 ^ x2) >> mask_bit
                    merge_idxs = idx1 + idx2
                    merged_list.append((merge_x, merge_idxs))
        return merged_list
    
    def compute_item(self, i: int, j: int):
        """ Compute the j-th item of the i-th list of messages.

        Args:
            i (int): the index of the list of messages
            j (int): the index of the message in the list
        
        Args:
            int: the hash value of the message
        """
        message = self.nonce + f"{i}-{j}".encode()
        return int.from_bytes(self.hashfunc(message), 'big')
    
    def compute_hash_list_on_the_fly(self, i: int):
        """ Compute the hash list on the fly.

        Args:
            i (int): the index of the list of messages
        
        Returns:
            List[int]: the list of hash values
        
        Note: In this case, the memory complexity is O(2^k N/2).
        """
        return [(self.compute_item(i, j), [j]) for j in range(2 ** self.mask_bit)]
    
    def compute_hash_list_on_the_fly(self, i: int, index_bit_length: int = None, index_value: int = None):
        """ Compute the hash list on the fly.

        Args:
            i (int): the index of the list of messages
            index_bit_length (int): the index bit length used in the first run
            index_value (int): the index value of the solution found in the first run
        
        Returns:
            List[int]: the list of hash values
        
        Note: In this case, the memory complexity is O(2^k N/2).
        """
        if index_bit_length is not None:
            if index_value is not None:
                # this is for the second run, so we need to save the full index with the index value
                return [(self.compute_item(i, j), [j]) for j in range(index_value, 2 ** self.mask_bit, 2 ** index_bit_length)]
            else:
                # this is for the first run, so we need to save all the indexes with bit length trimmed to `index_bit_length`
                return [(self.compute_item(i, j), [j % 2**index_bit_length]) for j in range(2 ** self.mask_bit)]
        else:
            # index_bit_length is None, return the full index
            return [(self.compute_item(i, j), [j]) for j in range(2 ** self.mask_bit)]
    
    def solve_on_the_fly(self, index_bit_length: int = None, index_vals: list[int] = None, verbose = False):
        """ Solve the generalized birthday problem on the fly.
        
        Args:
            index_bit_length (int): the index bit length used in the first run
            index_vals (list[int]): the index values of the solution found in the first run
            verbose (bool): whether to print the debug information
        
        Returns:
            List[(int, list)]: the list of indexes of messages such that XOR of their hash values is 0
        """
        # generate the binary tree root using post-order traversal
        index_vals = index_vals if index_vals is not None else [None] * self.k
        stack = deque([(self.compute_hash_list_on_the_fly(0, index_bit_length, index_vals[0]), 0)])
        
        for i in range(1, self.k):
            # check the top of the stack depth is the same as the current depth
            current_depth = 0
            merged_list = self.compute_hash_list_on_the_fly(i, index_bit_length, index_vals[i])
            while len(stack) != 0 and stack[-1][1] == current_depth:
                # pop the top of the stack
                top_hash_list, _ = stack.pop()
                # merge the top of the stack with the current hash_list
                if current_depth == self.lgk - 1:
                    merged_list = self.hash_join(top_hash_list, merged_list, self.mask_bit * 2)
                else:
                    merged_list = self.hash_join(top_hash_list, merged_list, self.mask_bit)
                if verbose: print(f"Merge two lists at depth {current_depth}, the length of merged list: {len(merged_list)}")
                current_depth += 1
            # push the current hash_list to the stack
            stack.append((merged_list, current_depth))
        # the root of the binary tree
        root_hash_list, depth = stack.pop()
        assert depth == self.lgk, f"The depth of the binary tree should be {self.lgk}"
        return [index_vec for _, index_vec in root_hash_list]
    
    def solve(self, index_bit_length: int = None, verbose = False):
        """ Solve the generalized birthday problem with generated message lists.
        
        Args:
            index_bit (int): The index bit length used in the first run (if this value is set, trade-off will be used).
            verbose (bool): whether to print the debug information
        
        Returns:
            List[(int, list)]: the list of indexes of messages such that XOR of their hash values is 0
        """
        if index_bit_length is None:
            # no trade-off, use the full-length index
            return self.solve_on_the_fly(verbose = verbose)
        else:
            # trade-off, use the index bit length
            # first run with trimmed index
            index_vectors = self.solve_on_the_fly(index_bit_length, verbose = verbose)
            # second run with reduced size of list
            # note: we can combine the index vectors to reduce the running times (might increase the initial list size).
            solutions = []
            for index_vector in index_vectors:
                if verbose: print(f"Second run with {index_vector = }")
                solutions += self.solve_on_the_fly(index_bit_length, index_vector, verbose = verbose)
        return solutions
                
    
def k_list_tester():
    # Test the WagnerSolver with the hash values computed on the fly
    n = 112
    k = 2 ** 6
    nonce = os.urandom(16)
    nonce = bytes.fromhex("58c3d5db02bd1617dc3e7844950d6f7c")
    print(f"{nonce.hex() = }")
    solver = KListSolver.new(n, k, nonce)
    st = time.time()
    
    root_hash_list = solver.solve(verbose=True)
    et = time.time()
    print(f"Time elapsed: {et - st:.2f} seconds")
    for idx in root_hash_list:
        # validate the result
        xor = 0
        # print(f"solution {idx = }")
        for i, j in enumerate(idx):
            xor ^= solver.compute_item(i, j)
        assert xor == 0, f"XOR of hash values should be 0, {xor = }"
    print(f"Found {len(root_hash_list)} messages such that XOR of their hash values is 0")
    
def k_list_tester_with_trade_off():
    # n = 112
    # k = 2 ** 6
    n = 200
    k = 2 ** 9
    nonce = os.urandom(16)
    # nonce = bytes.fromhex("58c3d5db02bd1617dc3e7844950d6f7c")
    print(f"{nonce.hex() = }")
    solver = KListSolver.new(n, k, nonce)
    
    st = time.time()
    root_hash_list = solver.solve(1, verbose=True)
    et = time.time()
    print(f"Time elapsed: {et - st:.2f} seconds")
    for idx in root_hash_list:
        # validate the result
        xor = 0
        for i, j in enumerate(idx):
            xor ^= solver.compute_item(i, j)
        assert xor == 0, f"XOR of hash values should be 0, {xor = }"
    print(f"Found {len(root_hash_list)} messages such that XOR of their hash values is 0")

if __name__ == "__main__":
    # k_list_tester()
    k_list_tester_with_trade_off()