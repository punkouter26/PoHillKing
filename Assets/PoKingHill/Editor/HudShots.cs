using System;
using System.Linq;
using System.Reflection;
using UnityEditor;
using UnityEngine;

namespace PoKingHill.EditorTools
{
    /// <summary>
    /// Game view sizing for HUD captures and aspect checks, driven from the Unity CLI:
    ///   unity command eval --code 'return PoKingHill.EditorTools.HudShots.SetGameView(1080, 1920);'
    /// Adds a fixed-resolution entry to the current size group when it is missing and selects it, so captures with
    /// capture_game_view show the layout a phone of that size gets (16:9 1080x1920, 19.5:9 1080x2340, 21:9 1080x2520).
    /// GameViewSizes and GameView are internal; everything goes through reflection and reports what failed.
    /// </summary>
    public static class HudShots
    {
        public static string SetGameView(int width, int height)
        {
            var asm = typeof(Editor).Assembly;
            var sizesType = asm.GetType("UnityEditor.GameViewSizes");
            var sizeType = asm.GetType("UnityEditor.GameViewSize");
            var sizeKind = asm.GetType("UnityEditor.GameViewSizeType");
            var gameViewType = asm.GetType("UnityEditor.GameView");
            if (sizesType == null || sizeType == null || sizeKind == null || gameViewType == null) return "GameView types not found";

            var instance = typeof(ScriptableSingleton<>).MakeGenericType(sizesType).GetProperty("instance").GetValue(null);
            var groupType = sizesType.GetProperty("currentGroupType").GetValue(instance);
            var group = sizesType.GetMethod("GetGroup").Invoke(instance, new[] { groupType });
            var gType = group.GetType();
            int count = (int)gType.GetMethod("GetTotalCount").Invoke(group, null);
            int index = -1;
            for (int i = 0; i < count; i++)
            {
                var s = gType.GetMethod("GetGameViewSize").Invoke(group, new object[] { i });
                if ((int)sizeType.GetProperty("width").GetValue(s) == width && (int)sizeType.GetProperty("height").GetValue(s) == height &&
                    (int)sizeType.GetProperty("sizeType").GetValue(s) == 1) { index = i; break; }
            }
            if (index < 0)
            {
                var ctor = sizeType.GetConstructors().First(c => c.GetParameters().Length == 4);
                var size = ctor.Invoke(new[] { Enum.ToObject(sizeKind, 1), width, height, $"KoTH {width}x{height}" });
                gType.GetMethod("AddCustomSize").Invoke(group, new[] { size });
                index = (int)gType.GetMethod("GetTotalCount").Invoke(group, null) - 1;
            }

            var view = EditorWindow.GetWindow(gameViewType, false, null, false);
            const BindingFlags F = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
            var prop = gameViewType.GetProperty("selectedSizeIndex", F);
            if (prop != null && prop.CanWrite) prop.SetValue(view, index);
            else gameViewType.GetMethod("SizeSelectionCallback", F)?.Invoke(view, new object[] { index, null });
            view.Repaint();
            return $"game view {width}x{height} (size {index}), screen {Screen.width}x{Screen.height}";
        }
    }
}
