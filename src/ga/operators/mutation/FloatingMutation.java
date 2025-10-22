package ga.operators.mutation;

import ga.core.Chromosome;
import ga.core.Gene;
import java.util.Random;

public class FloatingMutation implements MutationMethod {
    private final Random random = new Random();

    @Override
    public void mutate(Chromosome chromosome) {
        int index = random.nextInt(chromosome.getGenes().size());
        Gene g = chromosome.getGenes().get(index);

        double currentValue = g.getDoubleValue();
        double noise = (random.nextDouble() - 0.5) * 0.2; // ±10%
        g.setValue(currentValue + noise);
    }
}
