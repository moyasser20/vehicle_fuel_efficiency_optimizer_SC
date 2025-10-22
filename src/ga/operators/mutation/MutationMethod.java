package ga.operators.mutation;

import ga.core.Chromosome;

public interface MutationMethod {
    void mutate(Chromosome chromosome);
}
