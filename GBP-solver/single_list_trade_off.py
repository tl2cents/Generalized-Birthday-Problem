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
        
    def generate_hash_list(self, index_bit = None):
        """ Generate hash list.
        """
        single_list_size = self.N
        hash_list = []        
        # for j in trange(single_list_size, desc="Generating hash lists"):
        if index_bit is None:
            for j in range(single_list_size):
                hashval = self.compute_hash_item(j)
                hash_list.append(hashval)
        else:
            for j in range(single_list_size):
                hashval = self.compute_hash_item(j)
                hash_list.append((hashval, [j % (1<<index_bit)]))
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
    
    def hash_join_index_pointer(self, L: list, mask_bit: int, discard_zero=True) -> tuple[list, list]:
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
        merged_index = []
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
                if discard_zero == False and len(hash_table[x_low]) > 1:
                    # this is probably an invalid solution since collisions on the same item thrice is almost impossible.
                    # print(f"Detect Trivial Collisions For {x_low = }")
                    x1_high, idx1 = hash_table[x_low][0]
                    x2_high, idx2 = hash_table[x_low][1]
                    # remove the trivial collisions
                    idx = (idx2 , idx1)
                    if idx in merged_index:
                        assert x2_high ^ x1_high == 0
                        merged_list.remove(x2_high ^ x1_high)
                        merged_index.remove(idx)
                    continue
                for x2_high, j in hash_table[x_low]:
                    assert x2_high ^ x_high != 0 or (not discard_zero), f"{x2_high = }, {x_high = }"
                    merged_list.append(x2_high ^ x_high)
                    merged_index.append((i, j))
                    
                hash_table[x_low].append((x_high, i))
        return merged_list, merged_index
    
    def hash_join_index_vector(self, L: list, mask_bit: int, index_bit: int = None, check_table: dict = None, discard_zero=True) -> tuple[list, list]:
        """ Perform hash join on one list.
        Time complexity: O(n)
        Memory complexity: O(n)
        
        Args:
            L (List): the list of messages
            mask_bit (int): the number of bits to collide
        
        Returns:
            List[bytes]: the merged list of messages and indices
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
                if discard_zero and any(x2_high ^ x_high == 0 for x2_high, _ in hash_table[x_low]):
                    continue
                if discard_zero == False and len(hash_table[x_low]) > 1:
                    # this is probably an invalid solution since collisions on the same item thrice is almost impossible.
                    # print(f"Detect Trivial Collisions For {x_low = }")
                    x1_high, idx1 = hash_table[x_low][0]
                    x2_high, idx2 = hash_table[x_low][1]
                    # remove the trivial collisions
                    idx = idx1 + idx2
                    if check_table is not None:
                        idx = tuple(i % 2**index_bit for i in idx)
                    if (x2_high ^ x1_high, idx) in merged_list:
                        merged_list.remove((x2_high ^ x1_high, idx))
                    continue
                for x2_high, idx2 in hash_table[x_low]:
                    assert x2_high ^ x_high != 0 or (not discard_zero), f"{x2_high = }, {x_high = }"
                    if check_table is None:
                        merged_list.append((x2_high ^ x_high, idx1 +  idx2))
                    else:
                        idx = tuple(i % 2**index_bit for i in idx1 + idx2)
                        if idx in check_table:
                            merged_list.append((x2_high ^ x_high, idx1 + idx2))
                hash_table[x_low].append((x_high, idx1))
        return merged_list

    def solve_index_pointer(self, verbose = False):
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
            merged_list, current_idx = self.hash_join_index_pointer(merged_list, self.mask_bit)
            self.index_pointers.append(current_idx)
            if verbose: print(f"Round {i + 1}, the length of merged_list :{len(merged_list)}")
        merged_list, current_idx = self.hash_join_index_pointer(merged_list, self.mask_bit * 2, discard_zero=False)
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
                    # else:
                    #     if verbose: print(f"Duplicate Solution {k_prime = }")
                    if verbose: print(f"Solution found for {k_prime = }")
        if verbose: print(f"The length of solutions: {len(sols)}")
        self.index_pointers = []
        self.hash_list = None
        return sols
    
    def _solve_index_vector(self, index_bit: int = None, check_tables = None, verbose = False):
        """ Solve the generalized birthday problem on the fly.
        
        Args:
            verbose (bool): whether to print the debug information
        
        Returns:
            List[(int, list)]: the list of indexes of messages such that XOR of their hash values is 0
        """
        index_bit = index_bit if index_bit is not None else self.mask_bit + 1
        if check_tables is not None:
            self.generate_hash_list(self.mask_bit + 1)
        else:
            self.generate_hash_list(index_bit)
            check_tables = [None] * self.lgk
        index_bit = index_bit if index_bit is not None else self.mask_bit + 1
        merged_list = self.hash_list
        if verbose: print(f"Round 0, the length of merged_list :{len(merged_list)}")
        for i in range(self.lgk - 1):
            # if verbose: print(f"Round {i + 1}, {check_tables[i] = }")
            merged_list = self.hash_join_index_vector(merged_list, self.mask_bit, index_bit, check_tables[i], discard_zero=True)
            if verbose: print(f"Round {i + 1}, the length of merged_list :{len(merged_list)}")
        merged_list = self.hash_join_index_vector(merged_list, self.mask_bit * 2, index_bit, check_tables[self.lgk - 1], discard_zero=False)
        if verbose: print(f"Round {self.lgk}, the length of merged_list :{len(merged_list)}")
        return merged_list
    
    def solve_index_vector(self, index_bit: int = None, verbose = False):
        index_bit = index_bit if index_bit is not None else self.mask_bit + 1
        solu_cands = self._solve_index_vector(index_bit, verbose = verbose)
        solu_cands = set([tuple(index_vector) for val, index_vector in solu_cands])
        index_vectors = []
        if index_bit < self.mask_bit + 1:
            for index_vector in solu_cands:
                # we can also reserve the intermediate layer of height \hat{h}(k, t) to avoid multiple runs.
                check_tables = []
                if verbose: print(f"Second round run with constraint solution of {index_bit = }")
                # print(f"{index_vector = }")
                for h in range(1, self.lgk + 1):
                    l = 2**h
                    check_table = defaultdict(bool)
                    for i in range(0, len(index_vector), l):
                        check_table[tuple(index_vector[i:i+l])] = True
                    check_tables.append(check_table)
                index_vector = tuple(self._solve_index_vector(index_bit, check_tables, verbose)[0][1])
                index_vectors.append(index_vector)
        else:
            index_vectors = list(solu_cands)
        
        sols = []
        for index_vector in index_vectors:
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
                    if verbose: print(f"Solution found for {k_prime = }")
                else:
                    if verbose: print(f"Trivial solution {sorted(index_vector) = }")
        if verbose: print(f"The length of solutions: {len(sols)}")
        self.hash_list = None
        return sols
    
    def solve_index_vector_2_runs(self, index_bit: int = None, verbose = False):
        """
        When k is near the bound sqrt{n/2 + 1}, there might be many candidates. Denote the peak memory of the first run as M.
            - Ways to reduce the number of candidates:
                1. Filter out the trivial solutions before the final self-merge.
                2. Detect the trivial solutions in the final self-merge: multiple solutions from the same hash item are trivial solutions in the final self-merge.
                3. Increase the index_bit slightly greater than ell + 1 in the final self-merge. We can filter out all the trivial solutions just as the previous rounds.
            Methods 1,2 remove almost all the trivial solutions (method 3 is optional), but they are not guaranteed to remove all the non-trivial solutions.
            
            - Ways to speed up the second runs:
                1. We can merge the constraints into one table if the estimated memory peak does not exceed M.
                2. We can reserve the intermediate layer of height hat{h}(k, t) to speed up the second runs if the estimated memory peak of reserving two low-height layers does not exceed M.
        The proof-of-concept script will not cover optimal implementation details and just focus on the core idea of trade-off.
        """
        index_bit = index_bit if index_bit is not None else self.mask_bit + 1
        solu_cands = self._solve_index_vector(index_bit, verbose = verbose)
        solu_cands = set([tuple(index_vector) for val, index_vector in solu_cands])
        index_vectors = []
        if index_bit < self.mask_bit + 1:
            # we use dict to do the partial index check in constant time (there are other efficient ways to do this like splitting the input lists according to the partial solution).
            # Also, before the threshold height, this check can be ignored.
            check_tables = [defaultdict(bool) for _ in range(self.lgk)]
            for index_vector in solu_cands:
                if verbose: print(f"Second round run with constraint solution of {index_bit = }")
                # print(f"{index_vector = }")
                for h in range(1, self.lgk + 1):
                    l = 2**h
                    for i in range(0, len(index_vector), l):
                        check_tables[h - 1][tuple(index_vector[i:i+l])] = True
            res = self._solve_index_vector(index_bit, check_tables, verbose)
            index_vectors = [tuple(index_vector) for val, index_vector in res]
        else:
            index_vectors = list(solu_cands)
        sols = []
        for index_vector in index_vectors:
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
                    if verbose: print(f"Solution found for {k_prime = }")
                else:
                    if verbose: print(f"Trivial solution {sorted(index_vector) = }")
                    
        if verbose: print(f"The length of solutions: {len(sols)}")
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

def test_index_pointer():
    n = 160
    k = 9
    nonce = bytes.fromhex("41b7a058712f83aee32feaf4d02bc68a")    
    # nonce = os.urandom(16)
    print(f"{nonce.hex() = }")
    solver = OptimizedEquihashSolver(n, 2**k, nonce=nonce)
    print(f"n = {n}, k = {k}, {upper_bound(n) = } {n/(k+1) = }")
    results = solver.solve_index_pointer(verbose=True)
    solver.verify_results(results)

def test_index_vector_trade_off():
    n = 160
    k = 9
    nonce = bytes.fromhex("41b7a058712f83aee32feaf4d02bc68a")
    # nonce = os.urandom(16)
    print(f"{nonce.hex() = }")
    solver = OptimizedEquihashSolver(n, 2**k, nonce=nonce)
    print(f"n = {n}, k = {k}, {upper_bound(n) = } {n/(k+1) = }")
    results = solver.solve_index_vector(index_bit= 1, verbose=True)
    solver.verify_results(results)
    
def test_index_vector_2_runs():
    n = 160
    k = 9
    nonce = bytes.fromhex("41b7a058712f83aee32feaf4d02bc68a")
    # nonce = os.urandom(16)
    print(f"{nonce.hex() = }")
    solver = OptimizedEquihashSolver(n, 2**k, nonce=nonce)
    print(f"n = {n}, k = {k}, {upper_bound(n) = } {n/(k+1) = }")
    results = solver.solve_index_vector_2_runs(index_bit= 1, verbose=True)
    solver.verify_results(results)

if __name__ == "__main__":
    test_index_pointer()
    # test_index_vector_trade_off()
    # test_index_vector_2_runs()