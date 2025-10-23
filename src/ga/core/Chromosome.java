package ga.core;

import java.util.ArrayList;
import java.util.List;

public class Chromosome {
    private List<Gene> genes;
    private double fitness = Double.NEGATIVE_INFINITY;

    public Chromosome() {
        this.genes = new ArrayList<>();
    }

    public Chromosome(List<Gene> genes) {
        this.genes = new ArrayList<>(genes);
    }

    public List<Gene> getGenes() {
        return genes;
    }

    public void setGenes(List<Gene> genes) {
        this.genes = new ArrayList<>(genes);
    }

    public void addGene(Gene gene) {
        genes.add(gene);
    }

    public Gene getGene(int index) {
        return genes.get(index);
    }

    public void setGene(int index, Gene gene) {
        genes.set(index, gene);
    }

    public int getLength() {
        return genes.size();
    }

    public double getFitness() {
        return fitness;
    }

    public void setFitness(double fitness) {
        this.fitness = fitness;
    }

    public boolean getBinaryValue(int index) {
        return genes.get(index).getBooleanValue();
    }

    public int getIntegerValue(int index) {
        return genes.get(index).getIntegerValue();
    }

    public double getDoubleValue(int index) {
        return genes.get(index).getDoubleValue();
    }

    public Chromosome copy() {
        Chromosome copy = new Chromosome();
        for (Gene gene : genes) {
            copy.addGene(new Gene(gene.getValue()));
        }
        copy.setFitness(this.fitness);
        return copy;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Chromosome{fitness=").append(fitness).append(", genes=[");
        for (int i = 0; i < genes.size(); i++) {
            if (i > 0) sb.append(", ");
            sb.append(genes.get(i));
        }
        sb.append("]}");
        return sb.toString();
    }
}
