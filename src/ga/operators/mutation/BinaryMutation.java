package ga.operators.mutation;

import ga.core.Chromosome;
import ga.core.Gene;
import java.util.Random;

public class BinaryMutation implements MutationMethod {
    private final Random random = new Random();

    @Override
    public void mutate(Chromosome chromosome) {
        int index = random.nextInt(chromosome.getGenes().size());
        Gene g = chromosome.getGenes().get(index);

        boolean value = g.getBooleanValue();
        g.setValue(!value);
    }
}
