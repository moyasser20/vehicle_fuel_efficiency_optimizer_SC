package ga.operators.mutation;

import ga.core.Chromosome;
import ga.core.Gene;
import java.util.Random;

public class IntegerMutation implements MutationMethod {
    private final Random random = new Random();
    private final int min = 0;
    private final int max = 10;

    @Override
    public void mutate(Chromosome chromosome) {
        int index = random.nextInt(chromosome.getGenes().size());
        Gene g = chromosome.getGenes().get(index);

        g.setValue(min + random.nextInt(max - min + 1));
    }
}
