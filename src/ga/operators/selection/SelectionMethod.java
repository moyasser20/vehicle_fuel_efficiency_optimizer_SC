package ga.operators.selection;

import ga.core.Chromosome;
import ga.core.Population;
import java.util.List;


public interface SelectionMethod {
    List<Chromosome> select(Population population);
}
