using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApp2
{
    class Program
    {
        static void Main(string[] args)
        {
            var files = Directory.GetFiles(@"D:\alphapilot\AlphaPilot\GoodLabels");
            using (StreamWriter sw = new StreamWriter("vggFormatVal.json"))
            {
                sw.AutoFlush = true;
                sw.WriteLine("{");
                for (int i = 0; i < 2; i++)
                {
                    string fileText = File.ReadAllText(files[i]);
                    JObject o = JsonConvert.DeserializeObject(fileText) as JObject;
                    IList<string> keys = o.Properties().Select(p => p.Name).ToList();
                    for (int j = 0; j < keys.Count; j++)
                    {
                        try
                        {
                            o.TryGetValue(keys[j], out JToken value);
                            var x1 = value.First.First;
                            var y1 = x1.Next;
                            var x2 = y1.Next;
                            var y2 = x2.Next;
                            var x3 = y2.Next;
                            var y3 = x3.Next;
                            var x4 = y3.Next;
                            var y4 = x4.Next;
                            bool one=double.TryParse(x1.ToString(), out double x1d);
                            bool two=double.TryParse(x2.ToString(), out double x2d);
                            bool three=double.TryParse(x3.ToString(), out double x3d);
                            bool four=double.TryParse(x4.ToString(), out double x4d);
                            bool five=double.TryParse(y1.ToString(), out double y1d);
                            bool six=double.TryParse(y2.ToString(), out double y2d);
                            bool seven=double.TryParse(y3.ToString(), out double y3d);
                            bool eight=double.TryParse(y4.ToString(), out double y4d);
                            if (one && two && three && four && five && six && seven && eight)
                            {
                                sw.Write("\""+keys[j] + "\": ");
                                sw.WriteLine("{");
                                sw.WriteLine("\t\"filename\":\"" + keys[j] + "\",");
                                sw.WriteLine("\t\"regions\": {");
                                sw.WriteLine("\t\t\"0\": {");
                                sw.WriteLine("\t\t\t\"region_attributes\": {},");
                                sw.WriteLine("\t\t\t\"shape_attributes\": {");
                                sw.WriteLine("\t\t\t\t\"all_points_x\": [" + x1d+", "+x2d+", "+x3d+", "+x4d+" ],");
                                sw.WriteLine("\t\t\t\t\"all_points_y\": [" + y1d + ", " + y2d + ", " + y3d + ", " + y4d + " ],");
                                sw.WriteLine("\t\t\t\t\"name\":\"polygon\"");
                                sw.WriteLine("\t\t\t}");
                                sw.WriteLine("\t\t}");
                                sw.WriteLine("\t}");
                                sw.WriteLine("},");
                            }
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine(ex.Message);
                        }
                    }
                }
            }
        }
    }
}
