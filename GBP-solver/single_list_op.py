"""
Note: This is an optimized implementation of single-list algorithm using index pointer for the loose generalized birthday problem
where 'loose' means the the messages can come from a single list which is applicable to the Equihash mining algorithm.
"""

import hashlib
from math import ceil, floor, log2
import os
from collections import deque
from tqdm import trange
from collections import defaultdict

def blake2b(data : bytes, digest_byte_size : int = 64) -> bytes:
    return hashlib.blake2b(data, digest_size = digest_byte_size).digest()

class OptimizedEquihashSolver:
    """ 
    An optimized implementation of single-list algorithm for Equihash using index pointer!
    """
    def __init__(self, n, k, hashfunc = None, nonce = b"nonce"):
        """ Init a single-list solver for loose generalized birthday problem (Equihash) with parameters n, k. The underlying hash function is selected by `hashfunc` function.
        
        Loose generalized birthday problem: Given a hash function h with output length n, find k messages such that XOR of their hash values is 0.
        Args:
            n (int): the output bit length of hash function
            k (int): k messages to find, must be a power of 2
            hashfunc (function): the hash function to use
        """
        assert n % 8 == 0, "n should be a multiple of 8"
        if hashfunc is None:
            hashfunc = lambda x: blake2b(nonce + x, n // 8)
        assert len(hashfunc(b'')) == n // 8, "Hash function output length should be n/8 bytes"
        self.n = n
        self.k = k
        self.lgk = int(log2(k))
        assert 2 ** self.lgk == k, "k should be a power of 2"
        self.hashfunc = hashfunc
        self.hash_size = n // 8
        # self.mask_bit = (self.n + self.lgk) // (1 + self.lgk)
        # self.mask_bit = (self.n) // (1 + self.lgk)
        self.mask_bit = round(self.n / (1 + self.lgk))
        self.N = int(2 ** (self.n / (1 + self.lgk) + 1))
        self.mask = (1 << self.mask_bit) - 1
        self.mess_list = None
        self.hash_list = None
        self.index_pointers = []
        self.current_idx = None
        
    @staticmethod
    def new(n, k, hashfunc = None):
        return OptimizedEquihashSolver(n, k, hashfunc)
    
    def estimate(self) -> tuple[int, int]:
        """ Estimate the time and memory complexity to find k messages such that XOR of their hash values is 0.
    
        Returns:
            (int, int): the estimated time complexity and memory complexity
        """
        
        T = self.lgk * 2 ** (self.mask_bit + 1)
        M = 2*(self.n + self.lgk - self.mask_bit - 1) * 2 ** (self.mask_bit + 1)
        return T, M
    
    def compute_hash_item(self, i: int):
        """ Compute the i-th item of messages.

        Args:
            i (int): the index of messages        
        Args:
            int: the hash value of the message
        """
        message = f"message-{i}".encode()
        return int.from_bytes(self.hashfunc(message), 'big')
    
    def generate_message_list(self, message_len = None):
        """ Generate messages list randomly.
        """
        single_list_size = 2 ** (self.mask_bit + 1)
        if message_len is None:
            message_len = self.N

        mess_list = []
        hash_list = []        
        for j in trange(single_list_size, desc="Generating message lists"):
            # gnerate one list of messages
            message = os.urandom(message_len)
            hashval = int.from_bytes(self.hashfunc(message), 'big')
            mess_list.append(message)
            hash_list.append(hashval)
        self.mess_list = mess_list
        self.hash_list = hash_list
        
    def generate_hash_list(self):
        """ Generate hash list.
        """
        single_list_size = self.N
        hash_list = []        
        # for j in trange(single_list_size, desc="Generating hash lists"):
        for j in range(single_list_size):
            # gnerate one list of messages
            hashval = self.compute_hash_item(j)
            hash_list.append(hashval)
        self.hash_list = hash_list
        
    def get_current_index_vector(self, idx: int) -> set:
        """ Get the index vector of the current index.

        Args:
            idx (int): the current index point

        Returns:
            list: the index vector of the current index
        """
        idx_vec = [idx]
        for i in range(len(self.index_pointers) - 1, -1, -1):
            index_list = self.index_pointers[i]
            tmp = []
            for idx in idx_vec:
                idx1, idx2 = index_list[idx]
                tmp.append(idx1)
                tmp.append(idx2)
            idx_vec = tmp
        return idx_vec    
    
    def hash_join(self, L: list, mask_bit: int, discard_zero=True) -> tuple[list, list]:
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
        merged_idex = []
        mask = (1 << mask_bit) - 1

        for i, x1 in enumerate(L):
            x_low = x1 & mask
            x_high = x1 >> mask_bit
            if x_low not in hash_table:
                hash_table[x_low] = [(x_high, i)]
            else:
                if discard_zero and any(x2_high ^ x_high == 0 for x2_high, j in hash_table[x_low]):
                    # we discard the zero value because it is a trivial solution in most cases
                    continue
                for x2_high, j in hash_table[x_low]:
                    assert x2_high ^ x_high != 0 or (not discard_zero), f"{x2_high = }, {x_high = }"
                    merged_list.append(x2_high ^ x_high)
                    merged_idex.append((i, j))
                hash_table[x_low].append((x_high, i))
        return merged_list, merged_idex

    def solve(self, verbose = False):
        """ Solve the generalized birthday problem on the fly.
        
        Args:
            verbose (bool): whether to print the debug information
        
        Returns:
            List[(int, list)]: the list of indexes of messages such that XOR of their hash values is 0
        """
        # generate the binary tree root using postfix traversal
        self.generate_hash_list()
        merged_list = self.hash_list
        if verbose: print(f"Round 0, the length of merged_list :{len(merged_list)}")
        for i in range(self.lgk - 1):
            merged_list, current_idx = self.hash_join(merged_list, self.mask_bit)
            self.index_pointers.append(current_idx)
            if verbose: print(f"Round {i + 1}, the length of merged_list :{len(merged_list)}")
        merged_list, current_idx = self.hash_join(merged_list, self.mask_bit * 2, discard_zero=False)
        if verbose: print(f"Round {self.lgk}, the length of merged_list :{len(merged_list)}")
        self.index_pointers.append(current_idx)
        sols = []
        for i in range(len(current_idx)):
            index_vector = self.get_current_index_vector(i)
            set_weight = len(set(index_vector))
            if set_weight == self.k:
                sols.append(set(index_vector))
                k_prime = self.k
                if verbose: print(f"Solution found for {k_prime = }")
            else:
                # we might get some duplicate solutions i.e., the same message is used twice
                # this solution is also valid if we also allow solutions to LGBP(n, k-2) or LGBP(n, k-2i).
                counter = defaultdict(int)
                for idx in index_vector:
                    counter[idx] += 1 
                    counter[idx] %= 2
                # number of 1s in the counter
                k_prime = sum(counter.values())
                if k_prime != 0:
                    # not a trivial solution but still valid for LGBP(n, k_prime) i.e. LGBP(n, k-2i)
                    real_index_vector = set([idx for idx, val in counter.items() if val == 1])
                    if real_index_vector not in sols:
                        sols.append(real_index_vector)
                    else:
                        if verbose: print(f"Duplicate Solution {k_prime = }")
                    if verbose: print(f"Solution found for {k_prime = }")
        if verbose: print(f"The length of solutions: {len(sols)}")
        self.index_pointers = []
        self.hash_list = None
        return sols
    
    def solve_with_full_binary_tree(self, verbose = False):
        """ Solve the generalized birthday problem on the fly.
        
        Args:
            verbose (bool): whether to print the debug information
        
        Returns:
            List[(int, list)]: the list of indexes of messages such that XOR of their hash values is 0
        """
        # generate the binary tree root using postfix traversal
        self.generate_hash_list()
        merged_list = self.hash_list
        if verbose: print(f"Round 0, the length of merged_list :{len(merged_list)}")
        for i in range(self.lgk - 1):
            merged_list, current_idx = self.hash_join(merged_list, self.mask_bit)
            self.index_pointers.append(current_idx)
            if verbose: print(f"Round {i + 1}, the length of merged_list :{len(merged_list)}")
        merged_list, current_idx = self.hash_join(merged_list, self.mask_bit * 2, discard_zero=False)
        if verbose: print(f"Round {self.lgk}, the length of merged_list :{len(merged_list)}")
        self.index_pointers.append(current_idx)
        sols = []
        for i in range(len(current_idx)):
            index_vector = self.get_current_index_vector(i)
            sols.append(index_vector)
        if verbose: print(f"The length of solutions: {len(sols)}")
        self.index_pointers = []
        self.hash_list = None
        return sols
        
    def verify_results(self, result_indices):
        if len(result_indices) == 0:
            print("No solution found!")
            return
        for indices in result_indices:
            hashval = 0
            for idx in indices:
                hashval ^= self.compute_hash_item(idx)
            assert hashval == 0, f"{hashval = }"
        print("All solutions are correct!")


def upper_bound(n):
    return (n/2 + 1)**0.5

def count_solutions(n = 160, k = 9, indentifier = None, nsample=10000):
    # count the number of solutions for the loose generalized birthday problem
    import time
    import pickle
    if indentifier is None:
        indentifier = time.strftime("%Y%m%d-%H%M%S")
    log_file = open(f"{indentifier}_logfile_{n}_{k}.txt", "a+")
    log_file.write("%32s %10s %10s\n" % ("Nonce", "#Sol.", "Time/s"))
    total_solutions = 0
    sol_count = defaultdict(int)
    for i in trange(nsample):
        nonce = os.urandom(16)
        # print(f"Init {nonce.hex() = }")
        st = time.time()
        solver = OptimizedEquihashSolver(n, 2**k, nonce=nonce)
        results = solver.solve(strict=False, verbose=False)
        solver.verify_results(results)
        consu_time = time.time() - st
        # print(f"Time: {consu_time} s")
        # print(f"Number of solutions: {len(results)}")
        log_file.write("%32s %10d %10.6f\n" % (nonce.hex(), len(results), consu_time))
        total_solutions += len(results)
        for sol in results:
            sol_count[len(sol)] += 1
        if (i+1) % 100 == 0:
            print(f"Total number of solutions: {total_solutions}, Average number of solutions: {total_solutions/(i+1)}")
            # log the top 3 solutions with the highest frequency
            top_10 = sorted(sol_count.items(), key=lambda x: x[1], reverse=True)[:10]
            print(f"Top 10 solutions:")
            for j, (weight, freq) in enumerate(top_10):
                print(f"{j+1}: {weight = }, {freq = }, {freq/(i+1)}")
            # save the distribution of the number of solutions
            pickle.dump(sol_count, open(f"{indentifier}_sol_count_{n}_{k}.pkl", "wb"))
            print()
    log_file.write(f"Total number of solutions: {total_solutions}, Average number of solutions: {total_solutions/nsample}")

def test_full_tree():
    n = 48
    k = round(upper_bound(n))
    nonce = os.urandom(16)
    nonce = bytes.fromhex("31d3eaefdbb1e4c91311f64d32a60c0c")
    print(f"{nonce.hex() = }")
    solver = OptimizedEquihashSolver(n, 2**k, nonce=nonce)
    print(f"n = {n}, k = {k}, {upper_bound(n) = } {n/(k+1) = }")
    results = solver.solve_with_full_binary_tree(verbose=True)
    for result in results:
        print(result)
    solver.verify_results(results)


if __name__ == "__main__":
    count_solutions(96, 7, nsample=10000)
    count_solutions(160, 9, nsample=10000)
    