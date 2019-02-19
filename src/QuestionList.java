import java.net.Socket;
import java.util.*;

public class QuestionList {
    public static void main(String[]args) throws Exception {
        //testArrayListQueue();
        //test2DIterator();
        //testPreferenceList();
        //test10Wizards();
        //testFindMedian();
        //testBoggleGame();
        //testComMenu();
        //testRoundPrice();
        //testCSVParser();
        //testWebSocket();
        //TestGuessNumber();
        testWaterDrop();
    }

    private static void testWaterDrop() {
        int []input = new int[]{2,1,1,2,1,2,2};
        int []backup = Arrays.copyOf(input, input.length);
        new WaterDrop().pourWater(backup, 4, 3);
        int max = 0;
        for (int i : backup)
            max = Math.max(max, i);

        char[][]res = new char[max][input.length];
        for (int i = 0; i < max; i++) {
            int height = max - i;
            for (int j = 0; j < input.length; j++) {

                if (input[j] >= height) {
                    res[i][j] = '#';
                }
                else if (backup[j] >= height) {
                    res[i][j] = 'w';
                }
                else {
                    res[i][j] = ' ';
                }
            }

            System.out.println(new String(res[i]));
        }

    }

    private static void TestGuessNumber() {
        int worst = Integer.MAX_VALUE;
        List<String> list = new ArrayList<>();
        for (int i = 1; i <= 6; i++) {
            for (int j = 1; j <= 6; j++) {
                for (int k = 1; k <= 6; k++) {
                    for (int l = 1; l <= 6; l++) {
                        list.add(i + "" + j + "" +  k + "" +  l);
                    }
                }
            }
        }

        for (int t = 0; t < 20; t++) {
            for (String i : list) {
                var prob = new GuessNumber(Integer.parseInt(i));
                String res = prob.getBestCount();
                worst = Math.min(worst, prob.count);
            }
        }

        System.out.println(worst);
    }
    private static void testWebSocket() {
        Socket client = new Socket();

    }
    private static void testCSVParser() {
        System.out.println(new CSVParser().parseCSV("\"Alex\"\"A\"\"\",Med,am@g.com,1\"\"\"Alex a\"\"\""));
    }
    private static void testRoundPrice() {
        double []p = new double[]{1.2, 2.3, 3.4};
        int [] res = new RoundPrice().roundPrice(p);
        for (int rr : res)
            System.out.print(rr + ",");
    }
    private static void testComMenu() {
        System.out.println(new CombMenu().getCombinationSum(new double[]{1.2, 1.3, 3.4, 6.3, 0.3, 5.5}, 4.7));
    }
    private static void testBoggleGame() {
        Set<String> dict = new HashSet<>();
        dict.add("apple");
        dict.add("pear");
        dict.add("amer");
        char[][] matrix = new char[4][4];
        matrix[0] = "cbap".toCharArray();
        matrix[1] = "magp".toCharArray();
        matrix[2] = "epel".toCharArray();
        matrix[3] = "eear".toCharArray();

        List<String> res = new BoggleGame().findLongestPath(matrix, dict);
        System.out.println(res);
    }
    private static void testFindMedian() {
        double x = new FindMedian().findMedianWithBinarySearch(new int[]{1,1, 1, 2});
        System.out.println(x);
    }
    private static void test10Wizards() {
        List<List<Integer>> edges = new ArrayList<>();
        List<Integer> e0 = Arrays.asList(new Integer[]{1,2});
        List<Integer> e1 = Arrays.asList(new Integer[]{3});
        List<Integer> e2 = Arrays.asList(new Integer[]{3,4});
        List<Integer> e3 = Arrays.asList(new Integer[]{4});

        edges.add(e0);
        edges.add(e1);
        edges.add(e2);
        edges.add(e3);
        System.out.println(new TenWizards().getShortestPath(edges, 0, 4));
    }
    private static void testPreferenceList() {
        PreferenceList pl = new PreferenceList();
        pl.alienOrder(new String[]{"wrt", "wrf", "er", "ett", "rftt"});
    }
    private static void test2DIterator() {
        List<List<Integer>>test = new ArrayList<>();
        List<Integer> d1 = new LinkedList<>(Arrays.asList(new Integer[]{1,2,3,4,5}));
        List<Integer> d2 = new LinkedList<>(Arrays.asList(new Integer[]{6}));
        List<Integer> d3 = new LinkedList<>();
        List<Integer> d4 = new LinkedList<>(Arrays.asList(new Integer[]{7,8}));
        test.add(d1);
        test.add(d2);
        test.add(d3);
        test.add(d4);

        Flattern2DIterator it = new Flattern2DIterator(test);
        boolean isEven = true;
        while(it.hasNext()) {
            it.next();
            if(isEven) {
                it.remove();
            }

            isEven = !isEven;
        }

        it = new Flattern2DIterator(test);
        while(it.hasNext()) {
            System.out.println(it.next());
        }
    }
    public static void testArrayListQueue() throws Exception {
        ArrayListQueue queue = new ArrayListQueue(5);
        for (int i = 0; i < 100; i++) {
            queue.offer(i);
        }

        while(queue.size > 0) {
            System.out.println(queue.poll());
        }
    }
}




// 第一档
class WaterDrop{
    public int[] pourWater(int[] heights, int V, int K) {

        for (int v = 0; v < V; v++) {
            int leftDrop = findDirection(heights, K, -1);
            if (leftDrop > -1 && leftDrop != K) {
                heights[leftDrop]++;
                continue;
            }

            int rightDrop = findDirection(heights, K, 1);
            if (rightDrop < heights.length && rightDrop != K) {
                heights[rightDrop]++;
                continue;
            }

            heights[K]++;
        }

        return heights;
    }

    private int findDirection(int[] heights, int startIndex, int direction) {
        int lowestIndex = startIndex;
        for (int i = startIndex + direction; i >= 0 && i < heights.length; i += direction) {
            if (heights[i] > heights[i - direction]) {
                break; // can not flow over a peak
            }

            if (heights[i] < heights[lowestIndex]) {
                lowestIndex = i;
            }
        }

        return lowestIndex;
    }
}
class CheapestFlight{

    // an non-dp solution
    public int findCheapestPrice(int n, int[][] flights, int src, int dst, int K) {
        if (src == dst) return 0;
        int[][]table = new int[n][n];

        for (int[]f : flights) {
            table[f[0]][f[1]] = f[2];
        }

        PriorityQueue<Node> pq = new PriorityQueue<>(new Comparator<Node>() {
            @Override
            public int compare(Node o1, Node o2) {
                return o1.distance - o2.distance;
            }
        });

        pq.add(new Node(0, -1, src));

        while(!pq.isEmpty()) {
            Node cur = pq.remove();
            // Note: we only check on deque. This is the time we are sure about the distance
            if (cur.i == dst) {
                return cur.distance;
            }

            for (int next = 0; next < n; next++) {
                if (table[cur.i][next] == 0) continue;
                Node newNode = new Node(cur.distance + table[cur.i][next], cur.hop + 1, next);
                if (newNode.hop > K) continue;
                pq.add(newNode);
            }
        }

        return -1;
    }

    class Node{
        int distance;
        int hop;
        int i;
        public Node(int d, int hop, int i) {
            this.distance = d;
            this.hop = hop;
            this.i = i;
        }
    }
}
class FileSystem{
    Map<String, Integer> data;
    Map<String, Runnable> callbacks;
    public FileSystem(){
        this.data = new HashMap<>();
        this.callbacks = new HashMap<>();
    }

    public boolean create(String path, int value) {
        if (data.containsKey(path))
            return false;

        int lastIndex = path.lastIndexOf("/");
        if (lastIndex == -1) {
            return false;
        }

        String previousPath = path.substring(0, lastIndex);
        if (previousPath.length() == 0 || this.data.containsKey(previousPath)) {
            this.data.put(path, value);
            return true;
        }
        else {
            return false;
        }
    }

    public Integer get(String path) {
        return this.data.get(path);
    }

    public boolean set(String path, int value) {
        if (this.get(path) == null) return false;
        this.data.put(path, value);

        while(path.length() > 0) {
            if (callbacks.containsKey(path)) {
                callbacks.get(path).run();
            }

            int lastIndex = path.lastIndexOf("/");
            path = path.substring(0, lastIndex);
        }

        return true;
    }

    public void watch(String path, Runnable callback){
        if (this.get(path) != null) {
            this.callbacks.put(path, callback);
        }
    }
}
class PreferenceList{
    public String alienOrder(String[] words) {
        Map<Character, Set<Character>> higher = new HashMap<>();
        Map<Character, Set<Character>> lower = new HashMap<>();
        Set<Character> set = new HashSet<>();
        for (String w : words) {
            for (int i = 0; i < w.length() - 1; i++) {
                char c1 = w.charAt(i);
                char c2 = w.charAt(i+1);
                if (c1 == c2)
                    continue;
                if (higher.get(c2) == null) higher.put(c2, new HashSet<>());

                higher.get(c2).add(c1);

                if (lower.get(c1) == null) lower.put(c1, new HashSet<>());

                lower.get(c1).add(c2);

                set.add(c1);
                set.add(c2);
            }
        }

        Queue<Character> queue = new LinkedList<>();
        for (char c : set) {
            if (higher.get(c) == null || higher.size() == 0) {
                queue.add(c);
            }
        }

        StringBuilder sb = new StringBuilder();
        while(!queue.isEmpty()) {
            char c = queue.remove();
            sb.append(c);
            Set<Character> next = lower.get(c);
            if (next != null) {
                for (char cc : next) {
                    higher.get(cc).remove(c);
                    if (higher.get(cc).size() == 0) {
                        queue.add(cc);
                    }
                }
            }
        }

        return sb.length() == set.size() ?  sb.toString() : "";
    }
}

// 第二档
class SlidePuzzleSolver {
    public int slidingPuzzle(int[][] board) {
        Queue<String> queue = new LinkedList<>();
        String finalState = "123450";

        String initState = getState(board);
        queue.add(initState);
        Set<String> visit = new HashSet<>();
        visit.add(initState);

        int[]dxy = new int[]{-1,0,1,0,-1};
        int step = 0;
        while(!queue.isEmpty()) {
            int qsize = queue.size();
            for (int q = 0; q < qsize; q++) {
                String cur = queue.remove();
                if (cur.equals(finalState)) {
                    return step;
                }

                int emptyIndex = cur.indexOf("0");
                int i = emptyIndex/board[0].length;
                int j = emptyIndex%board[0].length;

                for (int d = 0; d < 4; d++) {
                    int newI = i + dxy[d];
                    int newJ = j + dxy[d+1];

                    if (newI < 0 || newI >= board.length || newJ < 0 || newJ >= board[0].length)
                        continue;

                    int newIndex = newI * board[0].length + newJ;
                    char[]array = cur.toCharArray();
                    swap(array, newIndex, emptyIndex);
                    String newState = new String(array);

                    if (visit.add(newState)) {
                        //  System.out.println(newState);
                        queue.add(newState);
                    }
                }
            }

            step++;
        }

        return -1;
    }

    private void swap(char[] arr, int i, int j) {
        char c = arr[i];arr[i] = arr[j]; arr[j] = c;
    }

    private String getState(int[][] board) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                sb.append(board[i][j]);
            }
        }

        return sb.toString();
    }
}
class PalindromePair{
    public List<List<Integer>> palindromePairs(String[] words) {
        List<List<Integer>> res = new ArrayList<>();
        Map<String, Integer> map = new HashMap<>();
        for (int i = 0; i < words.length; i++) {
            map.put(words[i], i);
        }

        for (int i = 0; i < words.length; i++) {
            String cur = words[i];
            String reverse = new StringBuilder(cur).reverse().toString();
            if (map.containsKey(reverse) && !cur.equals(reverse)) {
                List<Integer> temp = new ArrayList();
                temp.add(i);
                temp.add(map.get(reverse));
                res.add(temp);
            }

            if (cur.length() > 0 && isPalindrome(cur) && map.containsKey("")) {
                int index = map.get("");
                List<Integer> temp = new ArrayList();
                temp.add(i);
                temp.add(index);
                res.add(temp);

                temp = new ArrayList();
                temp.add(index);
                temp.add(i);
                res.add(temp);
            }

            for (int j = 1; j < cur.length(); j++) {
                if (isPalindrome(cur.substring(0, j))) {
                    String head = cur.substring(j);
                    String need = new StringBuilder(head).reverse().toString();
                    if (map.containsKey(need)) {
                        List<Integer> temp = new ArrayList<>();
                        temp.add(map.get(need));
                        temp.add(i);
                        res.add(temp);
                    }
                }

                if (isPalindrome(cur.substring(j))) {
                    String tail = cur.substring(0, j);
                    String tailneed = new StringBuilder(tail).reverse().toString();
                    if (map.containsKey(tailneed)) {
                        List<Integer> temp = new ArrayList<>();
                        temp.add(i);
                        temp.add(map.get(tailneed));
                        res.add(temp);
                    }
                }
            }
        }

        return res;
    }

    private boolean isPalindrome(String s) {
        String other = new StringBuilder(s).reverse().toString();
        return other.equals(s);
    }
}
class FindMedian{
    public double findMedianWithBinarySearch(int[]data){
        int max = findMax(data);
        int min = findMin(data);

        if (data.length%2 == 1) {
            return helper(data, min, max, data.length/2 + 1);
        }
        else {
            return (helper(data, min, max, data.length/2 + 1) + helper(data, min, max, data.length/2))*1.0/2;
        }
    }

    // rank means number of elements smaller or equal to itself
    private int helper(int[] data, int min, int max, int rank) {
        while(min < max) {
            int mid = (max - min)/2 + min;
            int maxRank = findRank(data, mid);
            if (maxRank >= rank) {
                max = mid;
            }
            else {
                min = mid + 1;
            }
        }

        return min;
    }

    private int findRank(int[] data, int mid) {
        int count = 0;
        for (int d : data) {
            if (d <= mid) count++;
        }

        return count;
    }

    private int findMin(int[] data) {
        int min = Integer.MAX_VALUE;
        for (int  d : data){
            min = Math.min(min, d);
        }

        return min;
    }

    private int findMax(int[] data) {
        int max = Integer.MIN_VALUE;
        for (int  d : data){
            max = Math.max(max, d);
        }

        return max;
    }
}
class Flattern2DIterator implements Iterator<Integer> {
    List<List<Integer>> data;
    List<Iterator<Integer>> iterators;
    int index;

    public Flattern2DIterator(List<List<Integer>> data){
        this.data = data;
        this.index = 0;
        iterators = new ArrayList<>();
        for (List<Integer> ls : data) {
            if (!ls.isEmpty())
                iterators.add(ls.iterator());
        }
    }

    @Override
    public boolean hasNext() {
        if (index >= iterators.size())
            return false;

        if (!iterators.get(index).hasNext()) {
            index++;
        }

        return (index < iterators.size() && iterators.get(index).hasNext());
    }

    @Override
    public Integer next() {
        int value = iterators.get(index).next();
        return value;
    }

    @Override
    public void remove() {
        iterators.get(index).remove();
    }
}
class CombMenu {
    public List<List<Integer>> getCombinationSum(double[]price, double target){
        List<List<Integer>> res = new ArrayList<>();
        for (int i = 0; i < price.length; i++) {
            dfs(price, i, target, res, new ArrayList<Integer>());
        }

        return res;
    }

    private void dfs(double[] price, int i, double target, List<List<Integer>> res, List<Integer> cur) {
        target -= price[i];
        cur.add(i);

        if (approxZero(target)) {
            res.add(new ArrayList<Integer>(cur));
        }
        else if (target > 0) {
            for (int j = i + 1; j < price.length; j++) {
                dfs(price, j, target, res, cur);
            }
        }

        cur.remove(cur.size() - 1);
    }

    private boolean approxZero(double target) {
        return target < 0.0001 && target > -0.0001;
    }
}
class CIDR{
    public List<String> ipToCIDR(String ip, int n) {
        long start = 0;
        for (String s : ip.split("\\.")) {
            start = start * 256 + Integer.parseInt(s);
        }

        List<String> res = new ArrayList<>();
        while(n > 0) {
            int num = Math.min((int)Long.lowestOneBit(start), Integer.highestOneBit(n));

            res.add(getCIDR(start, num));

            start += num;
            n -= num;
        }

        return res;
    }

    private String getCIDR(long x, int step) {
        int[] ans = new int[4];
        ans[0] = (int) (x & 255); x >>= 8;
        ans[1] = (int) (x & 255); x >>= 8;
        ans[2] = (int) (x & 255); x >>= 8;
        ans[3] = (int) x;
        int len = 32 - (int) (Math.log(step) / Math.log(2));
        return ans[3] + "." + ans[2] + "." + ans[1] + "." + ans[0] + "/" + len;
    }
}

class GuessNumber{
    int secret;
    int count = 0;
    public GuessNumber(int secret){
        this.secret = secret;
    }

    public String getBestCount(){
        List<String> list = new ArrayList<>();
        for (int i = 1; i <= 6; i++) {
            for (int j = 1; j <= 6; j++) {
                for (int k = 1; k <= 6; k++) {
                    for (int l = 1; l <= 6; l++) {
                        list.add(i + "" + j + "" +  k + "" +  l);
                    }
                }
            }
        }

        Collections.shuffle(list);

        while(list.size() > 1) {
            List<String> nextRound = new ArrayList<>();
            int[]res = guessAPI(list.get(0));
            count++;
            if (res[0] == 4) {
                return list.get(0);
            }

            for (int i = 1; i < list.size(); i++) {
                if (match(res, list.get(i), list.get(0))) {
                    nextRound.add(list.get(i));
                }
            }

            list = nextRound;
        }

        return list.iterator().next();
    }

    private boolean match(int[] res, String s, String s1) {
        String secrect = s1;
        int a = 0;
        int b = 0;

        int[]count = new int[6];
        int[]ownCount = new int[6];
        for (int i = 0; i < secrect.length(); i++) {
            if (secrect.charAt(i) == s.charAt(i)) {
                a++;
            }
            else {
                count[secrect.charAt(i) - '1']++;
                ownCount[s.charAt(i) - '1']++;
            }
        }

        for (int i = 0; i < 6; i++) {
            b += Math.min(count[i], ownCount[i]);
        }

        return a == res[0] && b == res[1];
    }

    private int[] guessAPI(String s) {
        String secrect = this.secret + "";
        int a = 0;
        int b = 0;

        int[]count = new int[6];
        int[]ownCount = new int[6];
        for (int i = 0; i < secrect.length(); i++) {
            if (secrect.charAt(i) == s.charAt(i)) {
                a++;
            }
            else {
                count[secrect.charAt(i) - '1']++;
                ownCount[s.charAt(i) - '1']++;
            }
        }

        for (int i = 0; i < 6; i++) {
            b += Math.min(count[i], ownCount[i]);
        }

        return new int[]{a,b};
    }
}

class BoggleGame{
    int h = 0;
    int w = 0;
    public List<String> findLongestPath(char[][]matrix, Set<String> dictonary) {
        h = matrix.length;
        w = matrix[0].length;

        List<List<int[]>> allValidPath = findAllValidSegment(matrix, dictonary);
        List<Integer> longestPathIndex = findLongPath(allValidPath);

        List<String> res = new ArrayList<>();
        for (Integer ls : longestPathIndex) {
            StringBuilder sb = new StringBuilder();
            for (int []id : allValidPath.get(ls)) {
                sb.append(matrix[id[0]][id[1]]);
            }

            res.add(sb.toString());
        }

        return res;
    }

    // dfs to find the longest path
    List<Integer> longestPath;

    private List<Integer> findLongPath(List<List<int[]>> allValidPath) {
        longestPath = new ArrayList<>();

        for (int i = 0; i < allValidPath.size(); i++)
            dfsFindLongestPath(allValidPath, new boolean[h][w], new LinkedList<Integer>(), i);

        return longestPath;
    }

    private void dfsFindLongestPath(List<List<int[]>> allValidPath, boolean[][] visit, LinkedList<Integer> curRes, int curIndex) {
        curRes.add(curIndex);
        if (curRes.size() > longestPath.size()) {
            longestPath = new ArrayList<>(curRes);
        }

        for (int i = 0; i < allValidPath.size(); i++) {
            if (canPut(allValidPath.get(i), allValidPath.get(curIndex), visit)) {
                updateVisit(visit, allValidPath.get(i));
                dfsFindLongestPath(allValidPath, visit, curRes, i);
                revertVisit(visit, allValidPath.get(i));
            }
        }

        curRes.remove(curRes.size() - 1);
    }

    private boolean canPut(List<int[]> cur, List<int[]> last, boolean[][] visit) {
        if (canPut(visit, cur)) {
            int[] curStart = cur.get(0);
            int[] lastEnd = last.get(last.size() - 1);
            if (isNb(curStart, lastEnd)) {
                return true;
            }
        }

        return false;
    }

    private boolean isNb(int[] curStart, int[] lastEnd) {
        int diff1= curStart[0] - lastEnd[0];
        int diff2 = curStart[1] - lastEnd[1];
        return Math.abs(diff1) + Math.abs(diff2) == 1;
    }

    private void revertVisit(boolean[][] visit, List<int[]> ints) {
        for (int []id : ints) {
            visit[id[0]][id[1]] = false;
        }
    }

    private void updateVisit(boolean[][] visit, List<int[]> ints) {
        for (int []id : ints) {
            visit[id[0]][id[1]] = true;
        }
    }

    private boolean canPut(boolean[][] visit, List<int[]> ints) {
        for (int []id : ints) {
            if (visit[id[0]][id[1]]) return false;
        }

        return true;
    }

    private List<List<int[]>> findAllValidSegment(char[][] matrix, Set<String> dictonary) {
        Trie trie = new Trie();
        for (String w : dictonary)
            trie.add(w);

        List<List<int[]>> res = new ArrayList<>();
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                dfsSearchWord(matrix, i, j, trie, res, "", new ArrayList<int[]>(), new boolean[h][w]);
            }
        }

        return res;
    }

    int[]dxy = new int[]{-1,0,1,0,-1};
    private void dfsSearchWord(char[][] matrix, int i, int j, Trie trie, List<List<int[]>> res, String prefix, ArrayList<int[]> cur, boolean[][]visit) {
        if (visit[i][j])
            return;

        prefix += matrix[i][j];
        visit[i][j] = true;
        cur.add(new int[]{i,j});

        if (!trie.containsPrefix(prefix)) {
            visit[i][j] = false;
            cur.remove(cur.size() - 1);
            return;
        }

        if (trie.containsWord(prefix)) {
            res.add(new ArrayList<>(cur));
        }

        for (int x = 0; x < 4; x++) {
            int newI = i + dxy[x];
            int newJ = j + dxy[x+1];
            if (newI < 0 || newI >= h || newJ < 0 || newJ >= w) continue;
            dfsSearchWord(matrix, newI, newJ, trie, res, prefix, cur, visit);
        }

        visit[i][j] = false;
        cur.remove(cur.size() - 1);
    }

    class Trie{
        TrieNode root = new TrieNode(false);

        public void add(String w) {
            TrieNode cur = root;
            for (int i = 0; i < w.length(); i++) {
                char c = w.charAt(i);
                if (cur.children[c-'a'] == null) {
                    cur.children[c-'a'] = new TrieNode(false);
                }

                if (i == w.length() - 1) {
                    cur.children[c-'a'].isWord = true;
                }

                cur = cur.children[c-'a'];
            }
        }

        public boolean containsPrefix(String prefix) {
            TrieNode cur = root;
            for (int i = 0; i < prefix.length(); i++) {
                char c = prefix.charAt(i);
                if (cur.children[c-'a'] == null) return false;
                cur = cur.children[c-'a'];
            }

            return true;
        }

        public boolean containsWord(String prefix) {
            TrieNode cur = root;
            for (int i = 0; i < prefix.length(); i++) {
                char c = prefix.charAt(i);
                if (cur.children[c-'a'] == null) return false;
                cur = cur.children[c-'a'];
            }

            return cur.isWord;
        }
    }
    class TrieNode{
        TrieNode[]children;
        boolean isWord;

        public TrieNode(boolean isWord){
            children = new TrieNode[26];
            this.isWord = isWord;
        }
    }
}

// 第三档
class WiggleSort{
    public void wiggleSort(int[] nums) {
        for (int i = 1; i < nums.length; i++) {
            if (i % 2 == 1) {
                if (nums[i] < nums[i - 1]) {
                    swap(nums, i, i - 1);
                }
            }
            else {
                if (nums[i] > nums[i-1]) {
                    swap(nums, i, i - 1);
                }
            }
        }
    }

    private void swap(int[] nums, int i, int i1) {
        int t = nums[i]; nums[i] = nums[i1]; nums[i1] = t;
    }

    public void wiggleSortII(int[] nums) {
        int median = findMedian(nums);
        int i = 0;
        int left = 0;
        int right = nums.length - 1;
        while(i <= right) {
            if (nums[mapIndex(i, nums.length)] > median) {
                swap(nums, mapIndex(i++, nums.length), mapIndex(left++, nums.length));
            }
            else if (nums[mapIndex(i, nums.length)] < median) {
                swap(nums, mapIndex(i, nums.length), mapIndex(right--, nums.length));
            }
            else {
                i++;
            }
        }
    }

    private int mapIndex(int i, int length) {
        return (2 * i + 1) % (length | 1);
    }

    private int findMedian(int[] nums) {
        // Quick Selection will make this O(n) and constance space
        Arrays.sort(nums);
        return nums[(nums.length - 1)/ 2];
    }
}
class ArrayListQueue{
    int arraySize;
    Node head;
    Node tail;
    List<Node> queue;
    int size;

    public ArrayListQueue(int fixSizeArray){
        this.arraySize = fixSizeArray;
        this.queue = new LinkedList<>();
        this.size = 0;
    }

    public void offer(int v){
        if (size == 0) {
            queue.add(new Node(this.arraySize));
            this.head = queue.get(0);
            this.tail = queue.get(0);
        }

        if (this.tail.curIndex == this.arraySize) {
            this.tail = new Node(this.arraySize);
            queue.add(this.tail);
        }

        this.size++;
        this.tail.array[this.tail.curIndex++] = v;
    }

    public int poll() throws Exception {
        if (this.size == 0)
            throw new Exception();

        if (this.head.removeIndex == this.arraySize) {
            this.queue.remove(0);
            this.head = this.queue.get(0);
        }

        this.size--;
        return this.head.array[this.head.removeIndex++];
    }

    private int[] getNewBlock(){
        return new int[arraySize];
    }

    class Node{
        int[]array;
        int curIndex;
        int removeIndex;
        public Node(int len){
            this.curIndex = 0;
            this.removeIndex = 0;
            this.array = new int[len];
        }
    }
}
class CollatzConjecture{
    Map<Integer, Integer> cache = new HashMap<>();
    public int findLongestSteps(int number){
        if (number < 1) return 0;

        int res = 0;
        for (int i = 1; i <= number; i++) {
            int curRes = findResult(i);
            cache.put(i, curRes);
            res = Math.max(curRes, res);
        }

        return res;
    }

    private int findResult(int i) {
        if (cache.containsKey(i)) return cache.get(i);

        if (i % 2 == 0) {
            return 1 + findResult(i/2);
        }
        else {
            return 1 + findResult(3*i+1);
        }
    }
}
class DisplayPages{
    public List<String> displayPages(List<String> input, int pageSize){
        List<String> res = new ArrayList<>();
        Iterator<String> iter = input.iterator();
        Set<String> idSet = new HashSet<>();
        boolean hasReachEnd = false;
        int counter = 0;
        while(iter.hasNext()) {
            String cur = iter.next();
            String id = cur.split(",")[0];
            if (!idSet.contains(id) || hasReachEnd) {
                res.add(cur);
                idSet.add(id);
                counter++;
                iter.remove();
            }

            if (counter == pageSize) {
                if (!input.isEmpty()) {
                    res.add(" ");

                    idSet.clear();
                    counter=0;
                    hasReachEnd = false;
                    iter = input.iterator();
                }
            }

            if (!iter.hasNext()) {
                hasReachEnd = true;
                iter = input.iterator();
            }
        }

        return res;
    }
}
// traval buddy
// minumum verties to travers

// 备用
class CSVParser{
    public String parseCSV(String str) {
        List<String> res = new ArrayList<>();
        boolean inQ = false;
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < str.length(); i++) {
            char c = str.charAt(i);
            if (inQ) {
                if (c != '"') {
                    sb.append(c);
                }
                else {
                    if (i == str.length() - 1 || str.charAt(i+1) != '"') {
                        inQ = false;
                    }
                    else {
                        sb.append('"');
                    }
                }
            }
            else {
                if (c == '"') {
                    inQ = true;
                }
                else if (c == ','){
                    res.add(sb.toString());
                    sb = new StringBuilder();
                }
                else {
                    sb.append(c);
                }

            }
        }

        if (sb.length() > 0) {
            res.add(sb.toString());
        }

        return String.join("|", res);
    }
}
class TenWizards{
    public List<Integer> getShortestPath(List<List<Integer>> wizards, int start, int end){
        Map<Integer, Integer> parents = new HashMap<>();
        Map<Integer, Integer> distance = new HashMap<>();
        distance.put(start, 0);

        PriorityQueue<Integer> queue = new PriorityQueue<>(new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return distance.get(o1) - distance.get(o2);
            }
        });

        queue.add(start);
        while(!queue.isEmpty()) {
            int cur = queue.remove();
            if (cur == end) {
                break;
            }

            int curDistance = distance.get(cur);
            List<Integer> nb = wizards.get(cur);
            for (Integer n : nb) {
                int newDistance = curDistance + getDistance(cur, n);

                if (distance.get(n) == null) {
                    parents.put(n, cur);
                    distance.put(n, newDistance);
                }
                else if (newDistance < distance.get(n)){
                    parents.put(n, cur);
                    distance.put(n, newDistance);
                }

                queue.remove(n);
                queue.add(n);
            }
        }

        if (distance.get(end) == null) {
            return new ArrayList<>(); //cannot reach
        }

        List<Integer> res = new LinkedList<>();
        res.add(end);

        while(end != start) {
            end = parents.get(end);
            res.add(0, end);
        }

        return res;
    }

    private int getDistance(int i, int j) {
        return (j-i) * (j-i);
    }
}
class RoundPrice{
    public int[] roundPrice(double[] prices) {
        double total = 0;
        for (double p : prices)
            total += p;

        int totalLowerRound = 0;
        for (double p : prices) {
            totalLowerRound += (int)p;
        }

        int target = (int)Math.round(total);
        int numberNeedsToLower = target - totalLowerRound;

        int[]res = new int[prices.length];
        PriorityQueue<Integer> pq = new PriorityQueue<>(new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                double diff1 = Math.ceil(prices[o1]) - prices[o1];
                double diff2 = Math.ceil(prices[o2]) - prices[o2];
                if (diff1 <= diff2) return -1;
                else return 1;
            }
        });

        for (int i = 0; i < res.length; i++)
        {
            res[i] = (int)prices[i];
            pq.add(i);
        }

        while(numberNeedsToLower > 0) {
            numberNeedsToLower--;
            int index = pq.remove();
            res[index]++;
        }

        return res;
    }
}


