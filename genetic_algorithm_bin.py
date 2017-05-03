import numpy as np


class GeneticAlgorithm:
    def __init__(self):
        self.population = np.empty(0)
        self.mutation_size = 0
        self.binary_length = 0
        self.scale = 1
        self.max, self.min = 0, 0

    def ga(self, init_population, max_generations, min_val, max_val, mutation_rate=0.1, crossover_rate = 0.1, precision=5):
        assert init_population > 0, max_generations >= 0
        assert max_val > min_val, precision >= 0
        assert 0.0 <= mutation_rate <= 1, 0.0 <= crossover_rate <= 1

        self.scale = 10 ** precision
        self.min, self.max = self.scale * min_val, self.scale * max_val
        self.population = np.array((np.random.uniform(min_val, max_val, init_population) * self.scale).astype(int))
        self.mutation_size = int(mutation_rate * self.population.size)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

        self.binary_length = np.log2(max(abs(self.min), abs(self.max))).astype(np.int) + 1
        tmp_max, tmp_argmax = 0.0, 0.0
        for i in range(max_generations):
            self.select()
            self.crossover()
            self.mutate()
            fn = self.fitness()
            tmp = max(fn)
            if tmp > tmp_max:
                tmp_max, tmp_argmax = tmp, self.population[np.argmax(fn)] / self.scale

        print(tmp_argmax, tmp_max)

    def fitness(self):
        # tmp = self.population[:, 0] + self.population[:, 1] / self.precision
        # return self.population + 10.0 * np.sin(5.0 * self.population) + 7.0 * np.cos(4.0 * self.population)
        tmp = self.population / self.scale
        return np.sin((tmp - 10000)) / (tmp - 10000)
        # pos = np.where(tmp != 10000)
        # tmp[pos] = np.sin(tmp[pos] - 10000) / (tmp[pos] - 10000)
        # return tmp
        # rst = np.empty_like(tmp)
        # for i in range(tmp.size):
        #     rst[i] = 0 if tmp[i] == 10000 else np.sin(tmp[i] - 1000) / (tmp[i] - 10000)
        # return rst
        # return np.sin(tmp) / (tmp+0.000001)

    def select(self):
        """
        Roulette wheel selection
        """
        fn = self.fitness()
        fn -= fn.min() - 1e-6  # make positive
        rnd = np.random.rand(len(fn))
        fn = np.append([-1], fn.cumsum() / fn.sum())  # cumulative_sum, normalized by sum
        count = [len([x for x in rnd if fn[i] < x <= fn[i + 1]]) for i in range(rnd.size)]
        self.population = np.concatenate([np.tile(self.population[i], (count[i], 1)) for i in range(len(count))])

    def crossover(self):
        np.random.shuffle(self.population)
        for i in range(self.population.size >> 1):
            l, r = i << 1, (i << 1) + 1
            lv, rv = self.population[l], self.population[r]
            tmpl, tmpr = self.population[l], self.population[r]
            for j in range(self.binary_length):
                lv, rv = (lv >> 1) & 0b1, (rv >> 1) & 0b1
                if lv != rv and np.random.uniform(0, 1) < self.crossover_rate:
                    tmpl, tmpr = tmpl ^ (1 << j), tmpr ^ (1 << j)
            self.population[l] = tmpl if self.min <= tmpl <= self.max else self.population[l]
            self.population[r] = tmpr if self.min <= tmpr <= self.max else self.population[r]
        #
        # pos = np.random.choice(np.arange(self.binary_length), self.population.size >> 1)
        # for i in range(pos.size):
        #     l, r, p = i << 1, (i << 1) + 1, pos[i]
        #     lv, rv = (self.population[l] >> p) & 0b1, (self.population[r] >> p) & 0b1
        #     if lv != rv:
        #         tmpl, tmpr = self.population[l] ^ (1 << p), self.population[r] ^ (1 << p)
        #         self.population[l] = tmpl if self.min <= tmpl <= self.max else self.population[l]
        #         self.population[r] = tmpr if self.min <= tmpr <= self.max else self.population[r]

    def mutate(self):
        # mutation position
        pos = np.random.choice(np.arange(self.binary_length), self.mutation_size)
        perm = np.random.permutation(self.population.size)[:self.mutation_size]
        for i in range(self.mutation_size):
            tmp = (self.population[i] ^ (1 << pos[i]))
            self.population[perm[i]] = tmp if self.min <= tmp <= self.max else self.population[perm[i]]


if __name__ == '__main__':
    import time

    g = GeneticAlgorithm()
    start_time = time.time()
    # np.random.seed(int(time.time()))
    g.ga(init_population=50, max_generations=50, min_val=0, max_val=1000000,
         mutation_rate=0.1, crossover_rate=0.1, precision=5)
    print("--- %s seconds ---" % (time.time() - start_time))
    # # 7.8570
