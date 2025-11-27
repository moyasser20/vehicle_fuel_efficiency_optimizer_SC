package fuzzy.core;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;


public class RuleBase implements Serializable {
    private static final long serialVersionUID = 1L;

    private final List<FuzzyRule> rules = new ArrayList<>();

    public RuleBase() {}

    public RuleBase(List<FuzzyRule> rules) { if (rules != null) this.rules.addAll(rules); }

    public synchronized FuzzyRule createRule() {
        FuzzyRule r = new FuzzyRule();
        rules.add(r);
        return r;
    }

    public synchronized void addRule(FuzzyRule r) { if (r != null) rules.add(r); }

    public synchronized boolean updateRule(FuzzyRule updated) {
        if (updated == null) return false;
        for (int i = 0; i < rules.size(); i++) {
            if (rules.get(i).getId().equals(updated.getId())) {
                rules.set(i, updated);
                return true;
            }
        }
        return false;
    }

    public synchronized boolean removeRule(String id) {
        return rules.removeIf(r -> r.getId().equals(id));
    }

    public synchronized boolean enableRule(String id, boolean enabled) {
        Optional<FuzzyRule> o = rules.stream().filter(r -> r.getId().equals(id)).findFirst();
        if (!o.isPresent()) return false;
        o.get().setEnabled(enabled);
        return true;
    }

    public synchronized boolean setWeight(String id, double weight) {
        Optional<FuzzyRule> o = rules.stream().filter(r -> r.getId().equals(id)).findFirst();
        if (!o.isPresent()) return false;
        o.get().setWeight(weight);
        return true;
    }

    public synchronized FuzzyRule getRule(String id) {
        return rules.stream().filter(r -> r.getId().equals(id)).findFirst().orElse(null);
    }

    public synchronized List<FuzzyRule> listRules() { return Collections.unmodifiableList(rules); }

    public synchronized void saveToFile(String path) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(path))) {
            oos.writeObject(rules);
        }
    }

    @SuppressWarnings("unchecked")
    public static RuleBase loadFromFile(String path) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(path))) {
            Object obj = ois.readObject();
            if (obj instanceof List) {
                return new RuleBase((List<FuzzyRule>) obj);
            }
            throw new IOException("File does not contain a rule list");
        }
    }
}
