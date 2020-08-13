import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.nio.file.Files;

public class getStats {

    public static void main(String[] args) {

        File folder = new File(args[0]);
        ArrayList<String> file_list = new ArrayList<String>();
        for (File file : folder.listFiles()) {
          String file_name = file.getName();
          if (!file_name.equals(".DS_Store")) file_list.add(file_name);
        }

        for (String file: file_list) {

          String file_header = file.split("\\_")[0];

          int min = -1;
          int max = -1;
          double median = -1;
          double first = -1;
          double third = -1;

          String file_name = folder+"/"+file;
          ArrayList<Integer> data = new ArrayList<Integer>();
          BufferedReader br = null;
          try {
              br = new BufferedReader(new FileReader(new File(file_name)));
              for (String line = br.readLine(); line != null; line = br.readLine()) {
                data.add(Integer.parseInt(line));
              }
          } catch(IOException e) {
              System.err.println("File '" + file_name +
                      "' not found or not readable.");
              System.exit(1);
          }
          Collections.sort(data);

          min = data.get(0);
          max = data.get(data.size()-1);
          median = getMedian(data);
          first = getFirst(data);
          third = getThird(data);

          try {
              String output_file_name = "stats/"+file_header+"/stats_"+file;
              File output_file = new File(output_file_name);
              FileWriter writer = new FileWriter(output_file);
              BufferedWriter bufferedWriter = new BufferedWriter(writer);
              bufferedWriter.write(Integer.toString(min));
              bufferedWriter.newLine();
              bufferedWriter.write(Integer.toString(max));
              bufferedWriter.newLine();
              bufferedWriter.write(Double.toString(median));
              bufferedWriter.newLine();
              bufferedWriter.write(Double.toString(first));
              bufferedWriter.newLine();
              bufferedWriter.write(Double.toString(third));
              bufferedWriter.close();
          } catch(IOException io) {
              System.err.println("Error reading the file:" + io);
              System.exit(1);
          }

        }
    }

    public static double getMedian(ArrayList<Integer> data) {

      if (data.size() % 2 == 1) {
          return data.get((data.size()+1)/2);
      } else {
          return (data.get(data.size()/2) + data.get((data.size()+2)/2)) / 2.0;
      }

    }

    public static double getFirst(ArrayList<Integer> data) {

      if (data.size() % 2 == 1) {
          int median = (data.size()+1)/2;
          if (median % 2 == 1) {
              return data.get((median+1)/2);
          } else {
            return (data.get(median/2) + data.get((median+2)/2)) /  2.0;
          }
      } else {
          int half = data.size()/2;
          return (data.get(half/2) + data.get((half+2)/2)) /  2.0;
      }
    }

      public static double getThird(ArrayList<Integer> data) {

        if (data.size() % 2 == 1) {
            int median = (data.size()+1)/2;
            if (median % 2 == 1) {
                return data.get(data.size()-((median+1)/2)+1);
            } else {
              return (data.get(data.size()-median/2) + data.get(data.size()-((median-2)/2)) /  2.0);
            }
        } else {
            int half = data.size()/2;
            return (data.get(data.size()-half/2) + data.get(data.size()-(half-2)/2) /  2.0);
        }

    }
}
