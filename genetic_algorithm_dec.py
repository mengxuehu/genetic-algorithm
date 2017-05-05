import numpy as np


class GeneticAlgorithm:
    def __init__(self):
        self.population = np.empty(0)
        self.population_str = np.empty(0)
        self.min, self.max = 0.0, 0.0
        self.len_integer, self.precision = 0, 5
        self.mutation_size = 0

    def ga(self, init_population, max_generations, min_val, max_val, mutation_rate=0.01, precision=5):
        assert init_population > 0, max_generations >= 0
        assert max_val > min_val, precision >= 0
        assert 0.0 <= mutation_rate <= 1
        self.population = np.random.uniform(min_val, max_val, init_population).reshape((init_population,))
        self.min, self.max = min_val, max_val
        self.mutation_size = int(mutation_rate * init_population)
        self.len_integer = (1 if np.min(self.population) < 0 else 0) + \
                           np.log10(max(abs(min_val), abs(max_val))).astype(int) + 1
        self.precision = precision
        y = np.zeros((max_generations,))
        for i in range(max_generations):
            self.select()
            np.random.shuffle(self.population)
            self.population_str = np.array(['{:0{l}.{p}f}'.format(x[0], l=self.len_integer + 1 + precision,
                                                                  p=precision) for x in self.population])
            self.crossover()
            self.mutate()
            self.population = np.array([float(x) for x in self.population_str])
            fn = self.fitness()
            y[i] = max(fn)
        return y

    def fitness(self):
        return np.sin(self.population - 10000) / (self.population - 10000)

    def select(self):
        """
        Roulette wheel selection
        """
        fn = self.fitness()
        fn -= fn.min()
        if fn.sum() == 0:
            return
        rnd = np.random.rand(len(fn))
        fn = np.append([-1], fn.cumsum() / fn.sum())  # cumulative_sum, normalized by sum
        count = [len([x for x in rnd if fn[i] < x <= fn[i + 1]]) for i in range(rnd.size)]
        self.population = np.concatenate([np.tile(self.population[i], (count[i], 1)) for i in range(len(count))])

    def crossover(self):
        np.random.shuffle(self.population)
        # crossover position
        pos = np.random.choice(np.arange(self.len_integer + self.precision), self.population.size >> 1)
        for i in range(pos.size):
            l, r, p = i << 1, (i << 1) + 1, pos[i]
            left, right = self.population_str[l], self.population_str[r]
            self.population_str[l] = left[:p] + right[p] + left[p + 1:]
            self.population_str[r] = right[:p] + left[p] + right[p + 1:]

    def mutate(self):
        # mutation position
        pos = np.random.choice(np.arange(self.len_integer + self.precision), self.mutation_size)
        target = np.random.choice(10, self.mutation_size).astype(str)
        perm = np.random.permutation(self.population.size)[:self.mutation_size]
        for j in range(perm.size):
            tmp = self.population_str[perm[j]]
            tmp = tmp[:pos[j]] + target[j] + tmp[pos[j] + 1:]
            self.population_str[perm[j]] = tmp if self.min <= float(tmp) <= self.max else self.population_str[perm[j]]


if __name__ == '__main__':
    g = GeneticAlgorithm()
    y = g.ga(init_population=100, max_generations=100, min_val=0, max_val=1000000, mutation_rate=1, precision=5).max()
    print(y)
