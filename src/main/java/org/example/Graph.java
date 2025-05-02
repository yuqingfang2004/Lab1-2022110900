package org.example;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;


// 此处为B1上的修改2
class Graph {
    //===================== 数据结构 =====================//
    private final Map<String, Map<String, Integer>> adjacencyList = new HashMap<>();
    private Map<String, Double> tfidfScores;

    //===================== 图构建 =====================//
    public void buildGraph(List<String> words) {
        adjacencyList.clear();
        tfidfScores = new HashMap<>();
        if (words.isEmpty()) return;

        // 计算词频(TF)
        Map<String, Integer> termFrequency = calculateTermFrequency(words);

        // 构建图结构
        buildGraphStructure(words);

        // 计算TF-IDF值
        calculateTfidfScores(termFrequency);
    }

    private Map<String, Integer> calculateTermFrequency(List<String> words) {
        Map<String, Integer> termFrequency = new HashMap<>();
        for (String word : words) {
            String lowerWord = word.toLowerCase();
            termFrequency.put(lowerWord, termFrequency.getOrDefault(lowerWord, 0) + 1);
        }
        return termFrequency;
    }

    private void buildGraphStructure(List<String> words) {
        for (int i = 0; i < words.size(); i++) {
            String current = words.get(i).toLowerCase();
            adjacencyList.putIfAbsent(current, new HashMap<>());

            if (i < words.size() - 1) {
                String next = words.get(i + 1).toLowerCase();
                addEdge(current, next);
            }
        }
    }

    private void calculateTfidfScores(Map<String, Integer> termFrequency) {
        int totalDocuments = 1; // 简化处理
        for (String word : adjacencyList.keySet()) {
            double tf = termFrequency.getOrDefault(word, 0);
            double df = 1; // 简化处理
            double idf = Math.log((totalDocuments + 1) / (df + 1)) + 1;
            tfidfScores.put(word, tf * idf);
        }
    }

    public void addEdge(String source, String target) {
        adjacencyList.computeIfAbsent(source, k -> new HashMap<>());
        adjacencyList.computeIfAbsent(target, k -> new HashMap<>());
        adjacencyList.get(source).merge(target, 1, Integer::sum);
    }

    //===================== 图查询 =====================//
    public boolean containsNode(String word) {
        return adjacencyList.containsKey(word.toLowerCase());
    }

    public Set<String> getSuccessors(String word) {
        return adjacencyList.getOrDefault(word.toLowerCase(), Collections.emptyMap()).keySet();
    }

    public boolean hasEdge(String source, String target) {
        return adjacencyList.getOrDefault(source.toLowerCase(), Collections.emptyMap())
                .containsKey(target.toLowerCase());
    }

    //===================== 桥接词 =====================//
    public String queryBridgeWords(String word1, String word2) {
        String w1 = word1.toLowerCase();
        String w2 = word2.toLowerCase();

        if (!containsNode(w1) || !containsNode(w2)) {
            return String.format("No \"%s\" or \"%s\" in the graph!", word1, word2);
        }

        List<String> bridges = getBridgeWords(w1, w2);
        return formatBridgeWordsOutput(word1, word2, bridges);
    }

    public List<String> getBridgeWords(String word1, String word2) {
        List<String> bridges = new ArrayList<>();
        for (String successor : getSuccessors(word1)) {
            if (hasEdge(successor, word2)) {
                bridges.add(successor);
            }
        }
        return bridges;
    }

    private String formatBridgeWordsOutput(String word1, String word2, List<String> bridges) {
        if (bridges.isEmpty()) {
            return String.format("No bridge words from \"%s\" to \"%s!\" ", word1, word2);
        }

        StringBuilder sb = new StringBuilder("The bridge words from \"")
                .append(word1).append("\" to \"").append(word2).append("\" are: \"");
        for (int i = 0; i < bridges.size(); i++) {
            if (i > 0) sb.append(i == bridges.size()-1 ? "\" and " : "\", ");
            sb.append(bridges.get(i)).append("\" ");
        }
        return sb.append(".").toString();
    }

    //===================== 最短路径 =====================//
    public String calcShortestPath(String word1, String word2) {
        PathResult result = shortestPath(word1, word2);
        if (result == null) {
            return String.format("\"%s\" 到 \"%s\" 不可达\n", word1, word2);
        }

        StringBuilder sb = new StringBuilder();
        // 显示第一条路径
        sb.append(String.format("最短路径: %s (总权重: %d)\n",
                String.join(" → ", result.paths.get(0)), result.totalWeight));

        // 显示其他路径
        if (result.paths.size() > 1) {
            sb.append(String.format("另外还有 %d 条最短路径:\n", result.paths.size() - 1));
            for (int i = 1; i < result.paths.size(); i++) {
                sb.append(String.format("  %d. %s\n", i, String.join(" → ", result.paths.get(i))));
            }
        }
        return sb.toString();
    }

    public PathResult shortestPath(String source, String target) {
        source = source.toLowerCase();
        target = target.toLowerCase();

        if (!containsNode(source) || !containsNode(target)) return null;

        Map<String, Integer> dist = new HashMap<>();
        Map<String, List<String>> prev = new HashMap<>();
        initializeShortestPathData(dist, prev);

        PriorityQueue<String> queue = new PriorityQueue<>(
                Comparator.comparingInt(node -> dist.getOrDefault(node, Integer.MAX_VALUE))
        );
        dist.put(source, 0);
        queue.add(source);

        executeDijkstra(target, dist, prev, queue);

        if (dist.get(target) == Integer.MAX_VALUE) return null;

        List<List<String>> paths = buildAllPaths(source, target, prev);
        return new PathResult(paths, dist.get(target));
    }

    private void initializeShortestPathData(Map<String, Integer> dist, Map<String, List<String>> prev) {
        adjacencyList.keySet().forEach(node -> {
            dist.put(node, Integer.MAX_VALUE);
            prev.put(node, new ArrayList<>());
        });
    }

    private void executeDijkstra(String target, Map<String, Integer> dist,
                                 Map<String, List<String>> prev, PriorityQueue<String> queue) {
        while (!queue.isEmpty()) {
            String u = queue.poll();
            if (u.equals(target)) break;

            for (Map.Entry<String, Integer> edge : adjacencyList.getOrDefault(u, Collections.emptyMap()).entrySet()) {
                relaxEdge(u, edge.getKey(), edge.getValue(), dist, prev, queue);
            }
        }
    }

    private void relaxEdge(String u, String v, int weight, Map<String, Integer> dist,
                           Map<String, List<String>> prev, PriorityQueue<String> queue) {
        int alt = dist.get(u) + weight;
        if (alt < dist.get(v)) {
            dist.put(v, alt);
            prev.get(v).clear();
            prev.get(v).add(u);
            queue.remove(v);
            queue.add(v);
        } else if (alt == dist.get(v)) {
            if (!prev.get(v).contains(u)) {
                prev.get(v).add(u);
            }
        }
    }

    private List<List<String>> buildAllPaths(String source, String target,
                                             Map<String, List<String>> prev) {
        List<List<String>> paths = new ArrayList<>();
        buildPaths(source, target, prev, new LinkedList<>(), paths);
        return paths;
    }

    private void buildPaths(String source, String current, Map<String, List<String>> prev,
                            LinkedList<String> path, List<List<String>> paths) {
        path.addFirst(current);
        if (current.equals(source)) {
            paths.add(new ArrayList<>(path));
        } else {
            for (String predecessor : prev.get(current)) {
                buildPaths(source, predecessor, prev, path, paths);
            }
        }
        path.removeFirst();
    }

    //===================== PageRank =====================//
    public Double calPageRank(String word) {
        // 固定参数：d=0.85, epsilon=0.0001, maxIter=100
        Map<String, Double> ranks = calcPageRank(0.85, 100);
        return ranks.getOrDefault(word.toLowerCase(), 0.0);
    }

    public Map<String, Double> calcPageRank(double dampingFactor, int maxIterations) {
        int N = adjacencyList.size();
        if (N == 0) return Collections.emptyMap();

        Map<String, Double> pageRank = new HashMap<>();
        initializePageRank(pageRank, N);

        Map<String, Integer> outDegree = calculateOutDegree();
        List<String> danglingNodes = identifyDanglingNodes(outDegree);

        pageRank = executePageRankIterations(dampingFactor, maxIterations, pageRank, outDegree, danglingNodes);
        return pageRank;
    }

    private void initializePageRank(Map<String, Double> pageRank, int totalNodes) {
        if (tfidfScores == null || tfidfScores.isEmpty()) {
            double initialValue = 1.0 / totalNodes;
            adjacencyList.keySet().forEach(node -> pageRank.put(node, initialValue));
            System.out.println("Using Average Initialize.");
            return;
        }

        DoubleSummaryStatistics stats = tfidfScores.values().stream()
                .collect(Collectors.summarizingDouble(Double::doubleValue));
        double minScore = stats.getMin();
        double maxScore = stats.getMax();
        double range = maxScore - minScore;

        for (String node : adjacencyList.keySet()) {
            double rawScore = tfidfScores.getOrDefault(node, minScore);
            double normalizedScore = (range < 1e-10) ? 1.0 :
                    0.5 + (rawScore - minScore) / range;
            pageRank.put(node, normalizedScore);
        }
        normalize(pageRank);
    }

    private Map<String, Integer> calculateOutDegree() {
        Map<String, Integer> outDegree = new HashMap<>();
        for (String node : adjacencyList.keySet()) {
            outDegree.put(node, adjacencyList.getOrDefault(node, Collections.emptyMap()).size());
        }
        return outDegree;
    }

    private List<String> identifyDanglingNodes(Map<String, Integer> outDegree) {
        return outDegree.entrySet().stream()
                .filter(e -> e.getValue() == 0)
                .map(Map.Entry::getKey)
                .collect(Collectors.toList());
    }

    private Map<String, Double> executePageRankIterations(double dampingFactor,
                                                          int maxIterations, Map<String, Double> pageRank,
                                                          Map<String, Integer> outDegree, List<String> danglingNodes) {
        int N = adjacencyList.size();
        for (int iter = 0; iter < maxIterations; iter++) {
            Map<String, Double> newRank = new HashMap<>();
            double df = dampingFactor;

            // 计算悬挂节点总和（使用当前pageRank值）
            double danglingSum = calculateDanglingContribution(danglingNodes, pageRank);

            // 计算每个节点的新PR值
            for (String node : adjacencyList.keySet()) {
                double sum = calculateRegularContribution(node, pageRank, outDegree);
                double value = (1 - df)/N + df * (sum + danglingSum/N);
                newRank.put(node, value);
            }

            pageRank = new HashMap<>(newRank);
        }
        return pageRank;
    }

    private double calculateRegularContribution(String node, Map<String, Double> currentRank,
                                                Map<String, Integer> outDegree) {
        double sum = 0;
        for (Map.Entry<String, Map<String, Integer>> entry : adjacencyList.entrySet()) {
            String v = entry.getKey();
            if (entry.getValue().containsKey(node)) {
                sum += currentRank.get(v) / outDegree.get(v);
            }
        }
        return sum;
    }

    private double calculateDanglingContribution(List<String> danglingNodes, Map<String, Double> currentRank) {
        return danglingNodes.stream()
                .mapToDouble(currentRank::get)
                .sum();
    }

    private void normalize(Map<String, Double> pageRank) {
        double total = pageRank.values().stream().mapToDouble(Double::doubleValue).sum();
        pageRank.replaceAll((k, v) -> v / total);
    }

    //===================== 随机游走 =====================//
    public String randomWalk() {
        // 线程安全数据结构
        AtomicBoolean stopFlag = new AtomicBoolean(false);
        List<String> path = Collections.synchronizedList(new ArrayList<>());
        Set<String> visitedEdges = Collections.synchronizedSet(new HashSet<>());

        // 启动游走线程
        Thread walkThread = new Thread(() -> {
            Random rand = new Random();
            String current = selectRandomStartNode(rand);
            if (current == null) return;
            path.add(current);

            while (!stopFlag.get()) {
                Map<String, Integer> edges = adjacencyList.get(current);
                if (edges == null || edges.isEmpty()) break;

                String next = selectNextNode(edges, rand);
                if (next == null) break;

                // 严格保持原有processEdge实现
                String edge = current + "->" + next;
                if (!visitedEdges.add(edge)) break; // 保持原有重复边检测逻辑

                path.add(next);
                current = next;

                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    break;
                }
            }
        });

        walkThread.start();

        // 中断监听（通过main函数处理I/O）
        try {
            while (walkThread.isAlive()) {
                if (System.in.available() > 0 && System.in.read() == 's') {
                    stopFlag.set(true);
                    walkThread.interrupt();
                    break;
                }
                Thread.sleep(100);
            }
        } catch (IOException | InterruptedException e) {
            // 异常处理
        }

        return String.join(" → ", path);
    }


    private String selectRandomStartNode(Random rand) {
        List<String> nodes = new ArrayList<>(adjacencyList.keySet());
        if (nodes.isEmpty()) return null;
        return nodes.get(rand.nextInt(nodes.size()));
    }


    private String selectNextNode(Map<String, Integer> edges, Random rand) {
        int totalWeight = edges.values().stream().mapToInt(Integer::intValue).sum();
        int randomPick = rand.nextInt(totalWeight);

        int cumulative = 0;
        for (Map.Entry<String, Integer> entry : edges.entrySet()) {
            cumulative += entry.getValue();
            if (randomPick < cumulative) {
                return entry.getKey();
            }
        }
        return null;
    }

    //===================== 图形输出相关 =====================//
    public String toTextGraph() {
        StringBuilder sb = new StringBuilder();
        for (String node : adjacencyList.keySet()) {
            sb.append(node).append(" -> ");
            Map<String, Integer> edges = adjacencyList.get(node);
            if (edges.isEmpty()) {
                sb.append("[]");
            } else {
                edges.forEach((target, weight) ->
                        sb.append(String.format("[%s(%d)] ", target, weight)));
            }
            sb.append("\n");
        }
        return sb.toString().trim();
    }

    public String toDOT() {
        StringBuilder sb = new StringBuilder("digraph G {\n");
        adjacencyList.forEach((source, targets) ->
                targets.forEach((target, weight) ->
                        sb.append(String.format("  \"%s\" -> \"%s\" [label=\"%d\"];\n",
                                source, target, weight))));
        sb.append("}");
        return sb.toString();
    }

    //===================== Getter方法 =====================//
    public Map<String, Map<String, Integer>> getAdjacencyList() {
        return Collections.unmodifiableMap(adjacencyList);
    }
}