package ga.operators.crossover;

import ga.core.Chromosome;
import ga.core.Gene;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class UniformCrossover implements CrossoverMethod {
    private final Random random = new Random();

    @Override
    public List<Chromosome> crossover(Chromosome p1, Chromosome p2) {
        int length = p1.getGenes().size();

        Chromosome c1 = new Chromosome();
        Chromosome c2 = new Chromosome();

        for (int i = 0; i < length; i++) {
            if (random.nextBoolean()) {
                c1.addGene(new Gene(p1.getGenes().get(i).getValue()));
                c2.addGene(new Gene(p2.getGenes().get(i).getValue()));
            } else {
                c1.addGene(new Gene(p2.getGenes().get(i).getValue()));
                c2.addGene(new Gene(p1.getGenes().get(i).getValue()));
            }
        }

        List<Chromosome> children = new ArrayList<>();
        children.add(c1);
        children.add(c2);
        return children;
    }
}
