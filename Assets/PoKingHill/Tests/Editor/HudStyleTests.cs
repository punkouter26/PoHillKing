using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using NUnit.Framework;

namespace PoKingHill.Tests
{
    /// <summary>
    /// The HUD's styling stays in one place: Hud.uss defines the type scale and control heights as tokens on :root and
    /// every other rule uses them, HudView sets no static sizes from C# (only seat colours, live widths, translations
    /// and show/hide), and every class HudView asks for exists in the stylesheet.
    /// </summary>
    public class HudStyleTests
    {
        const string Uss = "Assets/PoKingHill/UI/Hud.uss", View = "Assets/PoKingHill/Scripts/HudView.cs";

        static string Rules(out string root)
        {
            string uss = Regex.Replace(File.ReadAllText(Uss), @"/\*.*?\*/", "", RegexOptions.Singleline);
            var m = Regex.Match(uss, @":root\s*\{(.*?)\}", RegexOptions.Singleline);
            Assert.IsTrue(m.Success, "Hud.uss has a :root block");
            root = m.Groups[1].Value;
            return uss.Remove(m.Index, m.Length);
        }

        [Test]
        public void TypeSizesComeFromTheScale()
        {
            string rules = Rules(out string root);
            foreach (var t in new[] { "--fs-xs", "--fs-s", "--fs-m", "--fs-l", "--fs-xl", "--fs-xxl", "--h-s", "--h-m", "--h-l", "--h-xl" })
                StringAssert.Contains(t + ":", root, $"token {t} on :root");
            var raw = Regex.Matches(rules, @"font-size:\s*([^;]+);").Select(x => x.Groups[1].Value.Trim()).Where(v => !v.StartsWith("var(--fs-")).ToList();
            Assert.IsEmpty(raw, "font sizes outside :root use the scale: " + string.Join(", ", raw));
        }

        [Test]
        public void NoStaticSizesFromCode()
        {
            string code = File.ReadAllText(View);
            // sizes, margins and paddings belong in Hud.uss; code sets only live values (positions, percentages, insets)
            var bad = Regex.Matches(code, @"\.style\.(fontSize|height|minHeight|maxHeight|width|minWidth|maxWidth|margin\w*|padding\w*|flexGrow|flexShrink|flexBasis)\s*=\s*-?\d[^;]*;")
                           .Select(x => x.Value).ToList();
            Assert.IsEmpty(bad, "static style from HudView.cs:\n" + string.Join("\n", bad));
        }

        [Test]
        public void EveryClassTheHudUsesIsStyled()
        {
            string rules = Rules(out _);
            var defined = Regex.Matches(rules, @"\.([a-z][a-z0-9-]*)").Select(x => x.Groups[1].Value).ToHashSet();
            string code = File.ReadAllText(View);
            // class strings passed to El/Text/Btn/AddToClassList (two-word strings are split)
            var used = Regex.Matches(code, @"El\(""([a-z][a-z0-9 -]*)""|(?:Text|Btn)\((?:[^,""()]|""[^""]*""|\([^()]*\))*,\s*""([a-z][a-z0-9 -]*)""|AddToClassList\(""([a-z][a-z0-9-]*)""\)")
                            .SelectMany(x => (x.Groups[1].Success ? x.Groups[1].Value : x.Groups[2].Success ? x.Groups[2].Value : x.Groups[3].Value).Split(' '))
                            .Where(c => c.Length > 0).ToHashSet();
            var missing = used.Where(c => !defined.Contains(c)).OrderBy(c => c).ToList();
            Assert.IsEmpty(missing, "classes without a rule in Hud.uss: " + string.Join(", ", missing));
        }
    }
}
