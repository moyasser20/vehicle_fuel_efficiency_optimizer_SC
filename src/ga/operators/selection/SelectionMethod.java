package ga.operators.selection;

import ga.core.Chromosome;
import ga.core.Population;
import java.util.List;


//da interface lel select fel el3mom
public interface SelectionMethod {
    List<Chromosome> select(Population population);
}
