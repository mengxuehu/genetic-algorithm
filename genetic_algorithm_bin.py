import numpy as np


class GeneticAlgorithm:
    def __init__(self):
        self.population = np.empty(0)
        self.binary_length = 0
        self.scale = 1
        self.max, self.min = 0, 0
        self.mutation_rate, self.crossover_rate = 0.0, 0.0

    def ga(self, init_population, max_generations, min_val, max_val, mutation_rate=0.1, crossover_rate=0.1,
           precision=5):
        assert init_population > 0, max_generations >= 0
        assert max_val > min_val, precision >= 0
        assert 0.0 <= mutation_rate <= 1, 0.0 <= crossover_rate <= 1

        self.scale = 10 ** precision
        self.min, self.max = self.scale * min_val, self.scale * max_val
        self.population = np.array((np.random.uniform(min_val, max_val, init_population) * self.scale).astype(int))
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

        self.binary_length = np.log2(max(abs(self.min), abs(self.max))).astype(np.int) + 1

        y = np.zeros((max_generations,))
        for i in range(max_generations):
            self.select()
            self.crossover()
            self.mutate()
            fn = self.fitness()
            y[i] = max(fn)
        return y

    def fitness(self):
        tmp = self.population / self.scale
        return np.sin((tmp - 10000)) / (tmp - 10000)

    def select(self):
        """
        Roulette wheel selection
        """
        fn = self.fitness()
        fn = fn - fn.min()
        if fn.sum() == 0:
            return
        rnd = np.random.rand(len(fn))
        fn = np.append([-1], fn.cumsum() / fn.sum())  # cumulative_sum, normalized by sum
        count = [len([x for x in rnd if fn[i] < x <= fn[i + 1]]) for i in range(rnd.size)]
        self.population = np.concatenate([np.tile(self.population[i], (count[i], 1)) for i in range(len(count))])

    def crossover(self):
        np.random.shuffle(self.population)
        rnd = (np.random.uniform(0, 1, (self.population.size >> 1, self.binary_length)) < self.crossover_rate)
        for i in range(self.population.size >> 1):
            l, r = i << 1, (i << 1) + 1
            tmpl, tmpr = self.population[l], self.population[r]
            for j in range(self.binary_length):
                if rnd[i, j]:
                    lv, rv = (tmpl >> j) & 0b1, (tmpr >> j) & 0b1
                    if lv != rv:
                        tmpl ^= 1 << j
                        tmpr ^= 1 << j
            if self.min <= tmpl <= self.max:
                self.population[l] = tmpl
            if self.min <= tmpr <= self.max:
                self.population[r] = tmpr

    def mutate(self):
        rnd = (np.random.uniform(0, 1, (self.population.size, self.binary_length)) < self.mutation_rate)
        for i in range(self.population.size):
            tmp = self.population[i]
            for j in range(self.binary_length):
                if rnd[i, j]:
                    tmp ^= (1 << j)
            if self.min <= tmp <= self.max:
                self.population[i] = tmp


if __name__ == '__main__':
    g = GeneticAlgorithm()
    y = g.ga(init_population=100, max_generations=100, min_val=-1000000, max_val=1000000,
             mutation_rate=0.02, crossover_rate=0.3, precision=5).max()
    print(y)
