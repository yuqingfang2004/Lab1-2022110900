package org.example;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;


// 此处为B1以及C4分支上的修改2，手工消解冲突

class Graph {
    //===================== 数据结构 =====================//
    private final Map<String, Map<String, Integer>> adjacencyList = new HashMap<>();
    private Map<String, Double> tfidfScores;

    //===================== 图构建 =====================//

    /**
     * 根据给定的单词列表构建图结构，并计算TF-IDF分数。
     * 1. 清空邻接表和TF-IDF分数。
     * 2. 计算词频。
     * 3. 构建图结构。
     * 4. 计算TF-IDF值。
     *
     * @param words 单词列表
     */
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

    /**
     * 计算给定单词列表中每个单词的词频（TF）。
     *
     * @param words 单词列表
     * @return 包含单词及其词频的Map
     */
    private Map<String, Integer> calculateTermFrequency(List<String> words) {
        Map<String, Integer> termFrequency = new HashMap<>();
        for (String word : words) {
            String lowerWord = word.toLowerCase();
            termFrequency.put(lowerWord, termFrequency.getOrDefault(lowerWord, 0) + 1);
        }
        return termFrequency;
    }

    /**
     * 根据单词列表构建图的结构，添加节点和边。
     *
     * @param words 单词列表
     */
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

    /**
     * 根据词频计算每个单词的TF-IDF分数。
     *
     * @param termFrequency 包含单词及其词频的Map
     */
    private void calculateTfidfScores(Map<String, Integer> termFrequency) {
        int totalDocuments = 1; // 简化处理
        for (String word : adjacencyList.keySet()) {
            double tf = termFrequency.getOrDefault(word, 0);
            double df = 1; // 简化处理
            double idf = Math.log((totalDocuments + 1) / (df + 1)) + 1;
            tfidfScores.put(word, tf * idf);
        }
    }

    /**
     * 在图中添加一条从源节点到目标节点的边。
     * 如果节点不存在，则创建该节点。
     *
     * @param source 源节点
     * @param target 目标节点
     */
    public void addEdge(String source, String target) {
        adjacencyList.computeIfAbsent(source, k -> new HashMap<>());
        adjacencyList.computeIfAbsent(target, k -> new HashMap<>());
        adjacencyList.get(source).merge(target, 1, Integer::sum);
    }

    //===================== 图查询 =====================//

    /**
     * 检查图中是否包含指定的节点。
     *
     * @param word 要检查的节点
     * @return 如果包含则返回true，否则返回false
     */
    public boolean containsNode(String word) {
        return adjacencyList.containsKey(word.toLowerCase());
    }

    /**
     * 获取指定节点的所有后继节点。
     *
     * @param word 指定节点
     * @return 后继节点的集合
     */
    public Set<String> getSuccessors(String word) {
        return adjacencyList.getOrDefault(word.toLowerCase(), Collections.emptyMap()).keySet();
    }

    /**
     * 检查图中是否存在从源节点到目标节点的边。
     *
     * @param source 源节点
     * @param target 目标节点
     * @return 如果存在则返回true，否则返回false
     */
    public boolean hasEdge(String source, String target) {
        return adjacencyList.getOrDefault(source.toLowerCase(), Collections.emptyMap())
                .containsKey(target.toLowerCase());
    }

    //===================== 桥接词 =====================//

    /**
     * 查询从word1到word2的桥接词，并返回格式化的输出结果。
     * 如果word1或word2不在图中，则返回相应提示信息。
     *
     * @param word1 起始单词
     * @param word2 结束单词
     * @return 桥接词信息的字符串
     */
    public String queryBridgeWords(String word1, String word2) {
        if (word1 == null || word1.isEmpty() || word2 == null || word2.isEmpty()) {
            return "输入格式错误！";
        }
        String w1 = word1.toLowerCase();
        String w2 = word2.toLowerCase();

        if (!containsNode(w1) || !containsNode(w2)) {
            return String.format("No \"%s\" or \"%s\" in the graph!", word1, word2);
        }

        List<String> bridges = getBridgeWords(w1, w2);
        return formatBridgeWordsOutput(word1, word2, bridges);
    }

    /**
     * 获取从word1到word2的桥接词列表。
     * 桥接词是指word1的后继节点中，同时也是word2的前驱节点的单词。
     *
     * @param word1 起始单词
     * @param word2 结束单词
     * @return 桥接词列表
     */
    public List<String> getBridgeWords(String word1, String word2) {
        List<String> bridges = new ArrayList<>();
        for (String successor : getSuccessors(word1)) {
            if (hasEdge(successor, word2)) {
                bridges.add(successor);
            }
        }
        return bridges;
    }

    /**
     * 格式化桥接词的输出结果。
     * 如果没有桥接词，则返回相应提示信息；否则返回桥接词的详细信息。
     *
     * @param word1   起始单词
     * @param word2   结束单词
     * @param bridges 桥接词列表
     * @return 格式化后的桥接词信息字符串
     */
    private String formatBridgeWordsOutput(String word1, String word2, List<String> bridges) {
        if (bridges.isEmpty()) {
            return String.format("No bridge words from \"%s\" to \"%s!\" ", word1, word2);
        }

        StringBuilder sb = new StringBuilder("The bridge words from \"")
                .append(word1).append("\" to \"").append(word2).append("\" are: \"");
        for (int i = 0; i < bridges.size(); i++) {
            if (i > 0) sb.append(i == bridges.size() - 1 ? "\" and " : "\", ");
            sb.append(bridges.get(i)).append("\" ");
        }
        return sb.append(".").toString();
    }

    //===================== 最短路径 =====================//

    /**
     * 计算从word1到word2的最短路径，并返回格式化的输出结果。
     * 如果不存在路径，则返回相应提示信息。
     *
     * @param word1 起始单词
     * @param word2 结束单词
     * @return 最短路径信息的字符串
     */
    public String calcShortestPath(String word1, String word2) {
        StringBuilder sb = new StringBuilder();
        word1 = word1.toLowerCase();

        // 检查word1是否存在
        if (!containsNode(word1)) {
            sb.append("输入的单词有1或者2个不在图当中！");
            return sb.toString();
        }

        // 处理word2为空的情况（计算到所有节点的路径）
        if (word2 == null || word2.isEmpty()) {
            boolean hasPath = false;
            for (String target : adjacencyList.keySet()) {
                if (!target.equals(word1)) {
                    PathResult result = shortestPath(word1, target);
                    if (result != null) {
                        hasPath = true;
                        sb.append(String.format("到 \"%s\" 的最短路径:\n", target));
                        sb.append(String.format("  %s (总权重: %d)\n",
                                String.join(" → ", result.paths.get(0)), result.totalWeight));
                    }
                }
            }
            if (!hasPath) {
                sb.append(String.format("\"%s\" 到所有其他节点均不可达\n", word1));
            }
            return sb.toString();
        }

        // 处理word2不为空的情况（计算到指定节点的路径）
        word2 = word2.toLowerCase();
        if (!containsNode(word2)) {
            sb.append("输入的单词有1或者2个不在图当中！");
            return sb.toString();
        }

        PathResult result = shortestPath(word1, word2);
        if (result == null) {
            return String.format("\"%s\" 到 \"%s\" 不可达\n", word1, word2);
        }

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

    /**
     * 使用Dijkstra算法计算从源节点到目标节点的最短路径。
     *
     * @param source 源节点
     * @param target 目标节点
     * @return 包含最短路径列表和总权重的PathResult对象，如果不存在路径则返回null
     */
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

    /**
     * 初始化最短路径计算所需的数据，将所有节点的距离设为无穷大，前驱节点列表设为空。
     *
     * @param dist 存储节点到源节点距离的Map
     * @param prev 存储节点前驱节点列表的Map
     */
    private void initializeShortestPathData(Map<String, Integer> dist, Map<String, List<String>> prev) {
        adjacencyList.keySet().forEach(node -> {
            dist.put(node, Integer.MAX_VALUE);
            prev.put(node, new ArrayList<>());
        });
    }

    /**
     * 执行Dijkstra算法的核心逻辑，更新节点的距离和前驱节点信息。
     *
     * @param target 目标节点
     * @param dist   存储节点到源节点距离的Map
     * @param prev   存储节点前驱节点列表的Map
     * @param queue  用于存储待处理节点的优先队列
     */
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

    /**
     * 松弛边的操作，更新节点的距离和前驱节点信息。
     *
     * @param u      源节点
     * @param v      目标节点
     * @param weight 边的权重
     * @param dist   存储节点到源节点距离的Map
     * @param prev   存储节点前驱节点列表的Map
     * @param queue  用于存储待处理节点的优先队列
     */
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

    /**
     * 根据前驱节点信息构建从源节点到目标节点的所有最短路径。
     *
     * @param source 源节点
     * @param target 目标节点
     * @param prev   存储节点前驱节点列表的Map
     * @return 包含所有最短路径的列表
     */
    private List<List<String>> buildAllPaths(String source, String target,
                                             Map<String, List<String>> prev) {
        List<List<String>> paths = new ArrayList<>();
        buildPaths(source, target, prev, new LinkedList<>(), paths);
        return paths;
    }

    /**
     * 递归构建从源节点到当前节点的路径，并将完整路径添加到路径列表中。
     *
     * @param source  源节点
     * @param current 当前节点
     * @param prev    存储节点前驱节点列表的Map
     * @param path    当前构建的路径
     * @param paths   存储所有路径的列表
     */
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

    /**
     * 计算指定单词的PageRank值。
     * 使用固定参数：d=0.85, epsilon=0.0001, maxIter=100。
     *
     * @param word 要计算PageRank值的单词
     * @return 单词的PageRank值，如果单词不存在则返回0.0
     */
    public Double calPageRank(String word) {
        // 固定参数：d=0.85, epsilon=0.0001, maxIter=100
        Map<String, Double> ranks = calcPageRank(0.85, 100);
        return ranks.getOrDefault(word.toLowerCase(), 0.0);
    }

    /**
     * 计算图中所有节点的PageRank值。
     *
     * @param dampingFactor 阻尼因子
     * @param maxIterations 最大迭代次数
     * @return 包含节点及其PageRank值的Map
     */
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

    /**
     * 初始化PageRank值。
     * 如果TF-IDF分数不存在，则使用平均初始化；否则根据TF-IDF分数进行归一化初始化。
     *
     * @param pageRank   存储节点及其PageRank值的Map
     * @param totalNodes 节点总数
     */
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

    /**
     * 计算图中每个节点的出度。
     *
     * @return 包含节点及其出度的Map
     */
    private Map<String, Integer> calculateOutDegree() {
        Map<String, Integer> outDegree = new HashMap<>();
        for (String node : adjacencyList.keySet()) {
            outDegree.put(node, adjacencyList.getOrDefault(node, Collections.emptyMap()).size());
        }
        return outDegree;
    }

    /**
     * 识别图中的悬挂节点（出度为0的节点）。
     *
     * @param outDegree 包含节点及其出度的Map
     * @return 悬挂节点的列表
     */
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
                double value = (1 - df) / N + df * (sum + danglingSum / N);
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