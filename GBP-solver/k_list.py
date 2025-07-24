"""
Note: This is a basic implementation of Wagner's k-list algorithm for the strict generalized birthday problem
where 'strict' means the the messages must come from k distinct lists.
"""

import hashlib
from math import log2
import os
from collections import deque
from tqdm import trange

def blake2b(data : bytes, digest_byte_size : int = 64) -> bytes:
    return hashlib.blake2b(data, digest_size = digest_byte_size).digest()

class WagnerSolver:
    def __init__(self, n, k, hashfunc = None):
        """ Init a k-list solver for generalized birthday problem with parameters n, k. 
        The underlying hash function is selected by `hashfunc` function.
        
        Strict generalized birthday problem: Given a hash function h with output length n, 
        find k messages from pre-defined k distinct lists such that XOR of their hash values is 0.
        
        Args:
            n (int): the output bit length of hash function
            k (int): k messages to find, must be a power of 2
            hashfunc (function): the hash function to use
        """
        assert n % 8 == 0, "n should be a multiple of 8"
        if hashfunc is None:
            hashfunc = lambda x: blake2b(x, n // 8)
        assert len(hashfunc(b'')) == n // 8, "Hash function output length should be n/8 bytes"
        self.n = n
        self.k = k
        self.lgk = int(log2(k))
        assert 2 ** self.lgk == k, "k should be a power of 2"
        self.hashfunc = hashfunc
        self.hash_size = n // 8
        self.mask_bit = (self.n + self.lgk) // (1 + self.lgk) # ceil(n / (lgk + 1))
        self.mask = (1 << self.mask_bit) - 1
        
    @staticmethod
    def new(n, k, hashfunc = None):
        return WagnerSolver(n, k, hashfunc)
    
    def estimate(self) -> tuple[int, int]:
        """ Estimate the number of trials needed to find k messages such that XOR of their hash values is 0. Consider the optimal memory complexity with the k lists not stored in memory.
        
        Returns:
            (int, int): the estimated time complexity and memory complexity
        """
        N = 2 ** (self.mask_bit + 1)
        T = self.k * N
        M = (self.lgk + 2) * self.n * N / 4 + (2^(self.lgk-1) - 1) * self.mask_bit * N
        return T, M
    
    def generate_message_lists(self, message_len = None):
        """ Generate messages lists.
        """
        single_list_size = 2 ** (self.mask_bit)
        if message_len is None:
            message_len = ((self.lgk * self.mask_bit) * 2 + 7) // 8

        k_mess_lists = []
        k_hash_lists = []        
        for _ in trange(self.k, desc="Generating message lists"):
            # gnerate the k-th list of messages
            messages = []
            hashes = []
            for j in range(single_list_size):
                message = os.urandom(message_len)
                hashval = int.from_bytes(self.hashfunc(message), 'big')
                messages.append(message)
                hashes.append((hashval, [j]))
            k_mess_lists.append(messages)
            k_hash_lists.append(hashes)
        self.k_mess_lists = k_mess_lists
        self.k_hash_lists = k_hash_lists
        
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
        hash_table = {}
        mask = (1 << mask_bit) - 1
        
        for x1, idx1 in L1:
            if x1 & mask not in hash_table:
                hash_table[x1 & mask] = [(x1, idx1)]
            else:
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
    
    def solve_in_local(self, verbose = False):
        """ Solve the generalized birthday problem with generated message lists in local memory.
        
        Args:
            verbose (bool): whether to print the debug information
        
        Returns:
            List[(int, list)]: the list of indexes of messages such that XOR of their hash values is 0
        """
        assert self.k_hash_lists is not None, "Please generate message lists first"
        # generate the binary tree root using postfix traversal
        stack = deque([(self.k_hash_lists[0], 0)])
        for merged_list in self.k_hash_lists[1:]:
            # check the top of the stack depth is the same as the current depth
            current_depth = 0
            while len(stack) != 0 and stack[-1][1] == current_depth:
                # pop the top of the stack
                top_hash_list, top_depth = stack.pop()
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
        return root_hash_list
    
    def compute_item(self, i: int, j: int):
        """ Compute the j-th item of the i-th list of messages.

        Args:
            i (int): the index of the list of messages
            j (int): the index of the message in the list
        
        Args:
            int: the hash value of the message
        """
        message = f"{i}-{j}".encode()
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
    
    def solve_on_the_fly(self, verbose = False):
        """ Solve the generalized birthday problem on the fly.
        
        Args:
            verbose (bool): whether to print the debug information
        
        Returns:
            List[(int, list)]: the list of indexes of messages such that XOR of their hash values is 0
        """
        # generate the binary tree root using post-order traversal
        stack = deque([(self.compute_hash_list_on_the_fly(0), 0)])
        
        for i in range(1, self.k):
            # check the top of the stack depth is the same as the current depth
            current_depth = 0
            merged_list = self.compute_hash_list_on_the_fly(i)
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
        return root_hash_list
    
    def solve(self, on_the_fly = True, verbose = False):
        """ Solve the generalized birthday problem with generated message lists.
        
        Args:
            on_the_fly (bool): whether to compute the hash values on the fly
            verbose (bool): whether to print the debug information
        
        Returns:
            List[(int, list)]: the list of indexes of messages such that XOR of their hash values is 0
        """
        if on_the_fly:
            return self.solve_on_the_fly(verbose)
        else:
            return self.solve_in_local(verbose)

def high_memory_k_list_tester():
    # Test the WagnerSolver with the message lists generated in local memory
    n = 128
    k = 2 ** 7
    solver = WagnerSolver.new(n, k)
    solver.generate_message_lists()
    
    root_hash_list = solver.solve_in_local(verbose=True)
    for x, idx in root_hash_list:
        # validate the result
        xor = 0
        # print(f"solution {idx = }")
        for i, j in enumerate(idx):
            xor ^= solver.k_hash_lists[i][j][0]
        assert x==0 and xor == 0, f"XOR of hash values should be 0, {x = } {xor = }"
    print(f"Found {len(root_hash_list)} messages such that XOR of their hash values is 0")

    # n = 128 k = 2 ** 7
    # User time (seconds): 22.41
    # System time (seconds): 1.94
    # Percent of CPU this job got: 99%
    # Elapsed (wall clock) time (h:mm:ss or m:ss): 0:24.38
    # Average shared text size (kbytes): 0
    # Average unshared data size (kbytes): 0
    # Average stack size (kbytes): 0
    # Average total size (kbytes): 0
    # Maximum resident set size (kbytes): 2690476 -> 2GB
    
def low_memory_k_list_tester():
    # Test the WagnerSolver with the hash values computed on the fly
    n = 160
    k = 2 ** 9
    solver = WagnerSolver.new(n, k)
    T, M = solver.estimate()
    print(f"Estimated time complexity: 2^{log2(T)}, memory complexity: 2^{log2(M)}")    
    import time
    st = time.time()
    root_hash_list = solver.solve_on_the_fly(verbose=True)
    et = time.time()
    print(f"Time elapsed: {et - st:.2f} seconds")
    for x, idx in root_hash_list:
        # validate the result
        xor = 0
        # print(f"solution {idx = }")
        for i, j in enumerate(idx):
            xor ^= solver.compute_item(i, j)
        assert x==0 and xor == 0, f"XOR of hash values should be 0, {x = } {xor = }"
    print(f"Found {len(root_hash_list)} messages such that XOR of their hash values is 0")
    # n = 128 k = 2 ** 7
    # User time (seconds): 36.80
    # System time (seconds): 0.16
    # Percent of CPU this job got: 99%
    # Elapsed (wall clock) time (h:mm:ss or m:ss): 0:36.97
    # Average shared text size (kbytes): 0
    # Average unshared data size (kbytes): 0
    # Average stack size (kbytes): 0
    # Average total size (kbytes): 0
    # Maximum resident set size (kbytes): 234120 -> 228MB

if __name__ == "__main__":
    # high_memory_k_list_tester()
    low_memory_k_list_tester()