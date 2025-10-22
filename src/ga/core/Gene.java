package ga.core;

public class Gene {
    private Object value;

    public Gene(Object value) {
        this.value = value;
    }

    public Object getValue() {
        return value;
    }

    public void setValue(Object value) {
        this.value = value;
    }

    public boolean getBooleanValue() {
        if (value instanceof Boolean) {
            return (Boolean) value;
        }
        return false;
    }

    public int getIntegerValue() {
        if (value instanceof Number) {
            return ((Number) value).intValue();
        }
        return 0;
    }

    public double getDoubleValue() {
        if (value instanceof Number) {
            return ((Number) value).doubleValue();
        }
        return 0.0;
    }

    @Override
    public String toString() {
        return value.toString();
    }
}
