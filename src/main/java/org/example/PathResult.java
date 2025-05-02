package org.example;

import java.util.List;

public class PathResult {
    List<List<String>> paths;  // 所有最短路径
    int totalWeight;           // 路径权重

    public PathResult(List<List<String>> paths, int totalWeight) {
        this.paths = paths;
        this.totalWeight = totalWeight;
    }
}

