"""
Note: This is a basic implementation of single-list algorithm for the loose generalized birthday problem
where 'loose' means the the messages can come from a single list.
"""

import hashlib
from math import log2
import os
from collections import deque
from tqdm import trange

def blake2b(data : bytes, digest_byte_size : int = 64) -> bytes:
    return hashlib.blake2b(data, digest_size = digest_byte_size).digest()

class EquihashSolver:
    def __init__(self, n, k, hashfunc = None):
        """ Init a single-list solver for loose generalized birthday problem (Equihash) with parameters n, k. The underlying hash function is selected by `hashfunc` function.
        
        Loose generalized birthday problem: Given a hash function h with output length n, find k messages such that XOR of their hash values is 0.
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
        self.mess_list = None
        self.hash_list = None
        
    @staticmethod
    def new(n, k, hashfunc = None):
        return EquihashSolver(n, k, hashfunc)
    
    def estimate(self) -> tuple[int, int]:
        """ Estimate the time and memory complexity to find k messages such that XOR of their hash values is 0.
    
        Returns:
            (int, int): the estimated time complexity and memory complexity
        """
        
        T = self.lgk * 2 ** (self.mask_bit + 1)
        M = (self.k/2 * (self.mask_bit + 1) +  2*self.mask_bit) * 2 ** (self.mask_bit + 1)
        return T, M
    
    def generate_message_lists(self, message_len = None):
        """ Generate messages lists.
        """
        single_list_size = 2 ** (self.mask_bit + 1)
        if message_len is None:
            message_len = ((self.lgk * self.mask_bit) * 2 + 7) // 8

        mess_list = []
        hash_list = []        
        for j in trange(single_list_size, desc="Generating message lists"):
            # gnerate one list of messages
            message = os.urandom(message_len)
            hashval = int.from_bytes(self.hashfunc(message), 'big')
            mess_list.append(message)
            hash_list.append((hashval, set([j])))
        self.mess_list = mess_list
        self.hash_list = hash_list
        
    def hash_join(self, L: list, mask_bit: int) -> list:
        """ Perform hash join on one list.
        Time complexity: O(n)
        Memory complexity: O(n)
        
        Args:
            L (List): the list of messages
            mask_bit (int): the number of bits to collide
        
        Returns:
            List[bytes]: the merged list of messages and idexes
        """
        hash_table = {}
        merged_list = []
        mask = (1 << mask_bit) - 1
        for x1, idx1 in L:
            x_low = x1 & mask
            x_high = x1 >> mask_bit
            if x_low not in hash_table:
                hash_table[x_low] = [(x_high, idx1)]
            else:
                for x2, idx2 in hash_table[x_low]:
                    # chekc idx1 and idx2 do not overlap
                    if len(idx1 & idx2) == 0:
                        merged_list.append((x2 ^ x_high, idx1 | idx2))
                hash_table[x_low].append((x_high, idx1))
        return merged_list
    
    def solve_in_local(self, verbose = False):
        """ Solve the generalized birthday problem with generated message lists in local memory.
        
        Args:
            verbose (bool): whether to print the debug information
        
        Returns:
            List[(int, list)]: the list of indexes of messages such that XOR of their hash values is 0
        """
        assert self.hash_list is not None, "Please generate message lists first"
        # generate the binary tree root using postfix traversal
        merged_list = self.hash_list
        for i in range(self.lgk - 1):
            merged_list = self.hash_join(merged_list, self.mask_bit)
            print(f"Round {i + 1}, the length of merged_list :{len(merged_list)}")
        merged_list = self.hash_join(merged_list, self.mask_bit * 2)
        return merged_list
    
    def compute_item(self, i: int):
        """ Compute the i-th item of messages.

        Args:
            i (int): the index of messages        
        Args:
            int: the hash value of the message
        """
        message = f"message-{i}".encode()
        return int.from_bytes(self.hashfunc(message), 'big')
    
    def solve_on_the_fly(self, verbose = False):
        """ Solve the generalized birthday problem on the fly.
        
        Args:
            verbose (bool): whether to print the debug information
        
        Returns:
            List[(int, list)]: the list of indexes of messages such that XOR of their hash values is 0
        """
        N = 2 ** (self.mask_bit + 1)
        merged_list = [(self.compute_item(i), {i}) for i in range(N)]
        for i in range(self.lgk - 1):
            merged_list = self.hash_join(merged_list, self.mask_bit)
            print(f"Round {i + 1}, the length of merged_list :{len(merged_list)}")
        merged_list = self.hash_join(merged_list, self.mask_bit * 2)
        return merged_list
    
    
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

def tester_in_local():
    # Test the EquihashSolver
    n = 128
    k = 2 ** 7
    solver = EquihashSolver.new(n, k)
    T, M = solver.estimate()
    print(f"Estimated time complexity: 2^{log2(T)}, memory complexity: 2^{log2(M)}")


    solver.generate_message_lists()
    root_hash_list = solver.solve_in_local(verbose=True)
    for x, idx in root_hash_list:
        # validate the result
        assert x == 0 and len(idx) == k, f"XOR of hash values should be 0, {x = } {len(idx) = }"
        xor = 0
        # print(f"solution {idx = }")
        for i in idx:
            xor ^= solver.hash_list[i][0]
        assert xor == 0, f"XOR of hash values should be 0, {xor = }"
    print(f"Found {len(root_hash_list)} messages such that XOR of their hash values is 0")
    # n = 128 k = 2 ** 7
    #  User time (seconds): 3.79
    #  System time (seconds): 0.22
    #  Percent of CPU this job got: 99%
    #  Elapsed (wall clock) time (h:mm:ss or m:ss): 0:04.02
    #  Average shared text size (kbytes): 0
    #  Average unshared data size (kbytes): 0
    #  Average stack size (kbytes): 0
    #  Average total size (kbytes): 0
    #  Maximum resident set size (kbytes): 622840 -> 608MB
    
def tester_on_the_fly():
    # Test the EquihashSolver
    n = 128
    k = 2 ** 7
    solver = EquihashSolver.new(n, k)
    T, M = solver.estimate()
    print(f"Estimated time complexity: 2^{log2(T)}, memory complexity: 2^{log2(M)}")    

    root_hash_list = solver.solve_on_the_fly(verbose=True)
    for x, idx in root_hash_list:
        # validate the result
        assert x == 0 and len(idx) == k, f"XOR of hash values should be 0, {x = } {len(idx) = }"
        xor = 0
        # print(f"solution {idx = }")
        for i in idx:
            xor ^= solver.compute_item(i)
        assert xor == 0, f"XOR of hash values should be 0, {xor = }"
    print(f"Found {len(root_hash_list)} messages such that XOR of their hash values is 0")
    # n = 128 k = 2 ** 7
    # User time (seconds): 4.06
    # System time (seconds): 0.13
    # Percent of CPU this job got: 99%
    # Elapsed (wall clock) time (h:mm:ss or m:ss): 0:04.21
    # Average shared text size (kbytes): 0
    # Average unshared data size (kbytes): 0
    # Average stack size (kbytes): 0
    # Average total size (kbytes): 0
    # Maximum resident set size (kbytes): 557936 -> 544MB   

if __name__ == "__main__":
    # tester_in_local()
    tester_on_the_fly()