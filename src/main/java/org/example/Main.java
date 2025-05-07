package org.example;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;


// 这是git plugin的修改
class TextProcessor {
    /**
     * 处理文本文件，将其内容转换为小写，去除非字母字符，合并连续空格，并返回单词列表。
     *
     * @param file 待处理的文本文件
     * @return 处理后的单词列表
     * @throws IOException 读取文件时可能抛出的异常
     */
    public static List<String> processText(File file) throws IOException {
        String content = new String(Files.readAllBytes(file.toPath()), StandardCharsets.UTF_8);
        content = content.toLowerCase()
                .replaceAll("[^a-z]", " ")  // 保留字母和空格
                .replaceAll("\\s+", " ");   // 合并多个空格

        String[] words = content.split(" ");
        List<String> wordList = new ArrayList<>();
        for (String word : words) {
            if (!word.isEmpty()) {
                wordList.add(word);
            }
        }
        return wordList;
    }
}

// 这是在B2上的修改1
public class Main {
    /**
     * 展示有向图的文本结构，并可选择将其导出为图形文件。
     *
     * @param graph 要展示的有向图对象
     */
    private static void showDirectedGraph(Graph graph) {
        System.out.println("\n有向图结构：");
        System.out.println(graph.toTextGraph());
    }

    /**
     * 根据输入的文本和有向图，生成包含桥接词的新文本。
     *
     * @param graph     有向图对象
     * @param inputText 输入的文本
     * @return 生成的新文本
     */
    private static String generateNewText(Graph graph, String inputText) {
        // 分割并保留原始单词
        List<String> originalWords = new ArrayList<>();
        List<String> lowerWords = new ArrayList<>();

        String[] tokens = inputText.split("[^a-zA-Z]+");
        for (String token : tokens) {
            if (!token.isEmpty()) {
                originalWords.add(token);
                lowerWords.add(token.toLowerCase());
            }
        }

        if (originalWords.size() < 2) {
            return String.join(" ", originalWords);
        }

        // 构建新文本
        List<String> newWords = new ArrayList<>();
        newWords.add(originalWords.get(0));

        Random rand = new Random();
        for (int i = 0; i < originalWords.size() - 1; i++) {
            String current = lowerWords.get(i);
            String next = lowerWords.get(i + 1);

            // 查询桥接词
            List<String> bridges = graph.getBridgeWords(current, next);
            if (!bridges.isEmpty()) {
                // 随机选择桥接词
                String bridge = bridges.get(rand.nextInt(bridges.size()));
                newWords.add(bridge);
            }
            newWords.add(originalWords.get(i + 1));
        }

        return String.join(" ", newWords);
    }

    /**
     * 处理用户输入，调用generateNewText方法生成新文本并输出结果。
     *
     * @param scanner 用于读取用户输入的Scanner对象
     * @param graph   有向图对象
     */
    private static void handleNewTextGeneration(Scanner scanner, Graph graph) {
        System.out.print("\n请输入新文本（包含要处理的英文句子）: ");
        String input = scanner.nextLine();
        String result = generateNewText(graph, input);
        System.out.println("\n生成结果:\n" + result);
    }

    // 此处进行了第一次修改
    // 此处为B1上的修改1
    // 此处进行了第二次修改
    // 此处为C4分支上的修改1
    // 已经手工消解冲突

    /**
     * 程序的入口点，负责读取文本文件，构建有向图，并提供功能菜单供用户操作。
     *
     * @param args 命令行参数，可传入文本文件路径
     */
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        File file;
        try {
            // 处理文件路径输入
            if (args.length > 0) {
                file = new File(args[0]);
            } else {
                System.out.print("请输入文本文件路径: ");
                String path = scanner.nextLine().trim();
                file = new File(path);
            }

            List<String> words = TextProcessor.processText(file);
            System.out.println("成功处理单词数量: " + words.size());

            // 构建图结构
            Graph graph = new Graph();
            graph.buildGraph(words);
            System.out.println("图构建完成，包含节点数: " + graph.getAdjacencyList().size());

            while (true) {
                System.out.println("\n=== 功能菜单 ===");
                System.out.println("1. 展示有向图");
                System.out.println("2. 查询桥接词");
                System.out.println("3. 生成新文本");
                System.out.println("4. 计算最短路径");
                System.out.println("5. 计算PageRank");
                System.out.println("6. 执行随机游走");
                System.out.println("7. 退出");
                System.out.print("请选择操作: ");

                String choice;
                synchronized (System.in) {
                    choice = scanner.nextLine().trim();
                }
                switch (choice) {
                    case "1":
                        showDirectedGraph(graph);
                        System.out.print("\n是否需要生成图形文件？(y/n) ");
                        if (scanner.nextLine().trim().equalsIgnoreCase("y")) {
                            try {
                                GraphExporter.exportToImage(graph, "png");
                            } catch (Exception e) {
                                System.err.println("图形导出错误: " + e.getMessage());
                            }
                        }
                        break;
                    case "2":
                        System.out.println("\n=== 桥接词查询 ===");
                        Scanner scanner_query = new Scanner(System.in);
                        while (true) {
                            System.out.print("输入两个单词（空格分隔/q退出）: ");
                            String input = scanner_query.nextLine().trim();
                            if (input.equalsIgnoreCase("q")) break;

                            String[] query_words = input.split("\\s+");
                            if (query_words.length != 2) {
                                System.out.println("输入格式错误！");
                                continue;
                            }
                            System.out.println(graph.queryBridgeWords(query_words[0], query_words[1]));
                        }
                        break;
                    case "3":
                        handleNewTextGeneration(scanner, graph);
                        break;
                    case "4":
                        System.out.print("输入一个或两个单词（空格分隔）: ");
                        String[] bridge_query_words = scanner.nextLine().trim().split("\\s+");

                        if (bridge_query_words.length == 1) {
                            // 单个单词：计算到所有节点的路径
                            String source = bridge_query_words[0];
                            if (!graph.containsNode(source.toLowerCase())) {
                                System.out.println("图中不包含该单词");
                                break;
                            }
                            StringBuilder allPaths = new StringBuilder();
                            for (String target : graph.getAdjacencyList().keySet()) {
                                if (!source.equalsIgnoreCase(target)) {
                                    allPaths.append(graph.calcShortestPath(source, target));
                                }
                            }
                            System.out.println(allPaths);
                        } else if (bridge_query_words.length == 2) {
                            if (!graph.containsNode(bridge_query_words[0].toLowerCase()) || !graph.containsNode(bridge_query_words[1].toLowerCase())) {
                                System.out.println("输入的单词有1或者2个不在图当中！");
                                break;
                            }
                            // 两个单词：计算指定路径
                            System.out.println(graph.calcShortestPath(bridge_query_words[0], bridge_query_words[1]));
                        } else {
                            System.out.println("输入格式错误！");
                        }
                        break;
                    case "5":
                        System.out.print("输入要查询的单词（直接回车显示全部）: ");
                        String word = scanner.nextLine().trim();

                        if (word.isEmpty()) {
                            // 批量查询模式
                            System.out.println("\n所有单词的PageRank值（d=0.85）：");

                            // 遍历所有节点但保持calPageRank封装不变
                            graph.getAdjacencyList().keySet().stream()
                                    .map(w -> new AbstractMap.SimpleEntry<>(w, graph.calPageRank(w)))
                                    .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
                                    .forEach(entry -> System.out.printf("%-15s %.6f\n", entry.getKey(), entry.getValue()));
                        } else {
                            // 单个单词查询
                            Double prValue = graph.calPageRank(word);
                            System.out.printf("单词 \"%s\" 的PageRank值（d=0.85）: %.6f\n", word, prValue);
                        }
                        break;
                    case "6":
                        System.out.println("\n=== 随机游走 ===");
                        System.out.println("开始随机游走，输入's'停止...");

                        // 调用规范接口（内部已包含中断逻辑）
                        String walkPath = graph.randomWalk();

                        try {
                            while (System.in.available() > 0) {
                                System.in.read();
                            }
                        } catch (IOException e) {
                            ;
                        }

                        // 处理结果
                        System.out.println("最终路径: " + walkPath);
                        try {
                            Path path = Paths.get("./src/test/random_walk_" +
                                    LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss")) + ".txt");
                            Files.write(path, walkPath.getBytes());
                            System.out.println("结果已保存至: " + path.toAbsolutePath());
                        } catch (IOException e) {
                            System.err.println("保存失败: " + e.getMessage());
                        }
                        break;
                    case "7":
                        System.out.println("程序已退出");
                        return;
                    default:
                        System.out.println("无效输入！");
                }
            }
        } catch (IOException e) {
            System.err.println("文件处理错误: " + e.getMessage());
        }
    }
}