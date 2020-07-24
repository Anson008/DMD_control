import numpy as np


class PeriodGenerator:
    @staticmethod
    def prime_numbers():
        prime_num = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
                     31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
                     73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
                     127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
                     179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
                     233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
                     283, 293, 307, 311, 313, 317, 331, 337, 347, 349,
                     353, 359, 367, 373, 379, 383, 389, 397, 401, 409,
                     419, 421, 431, 433, 439, 443, 449, 457, 461, 463,
                     467, 479, 487, 491, 499, 503, 509, 521, 523, 541]
        return prime_num

    @staticmethod
    def prime_number_list(upper):
        # Initialize a list
        primes = []
        for candidate in range(2, upper + 1):
            # Assume number is prime until shown it is not.
            is_prime = True
            for num in range(2, int(candidate ** 0.5) + 1):
                if candidate % num == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(candidate)
        return primes

    @staticmethod
    def save_to_csv(prime_list, path):
        np.savetxt(path, prime_list, fmt='%d', delimiter=',')


if __name__ == "__main__":
    periods = PeriodGenerator()
    prime_list = periods.prime_number_list(3000)
    print(len(prime_list))
    print(prime_list)

    periods.save_to_csv(prime_list, 'prime_numbers_lt_3000.csv')
