using System.IO;
using NUnit.Framework;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace PoKingHill.Tests
{
    /// <summary>Gate B.2: no PhysX anywhere in PoKingHill scenes, and PhysX auto-simulation is off project-wide.</summary>
    public class PhysXAudit
    {
        static string[] Scenes() =>
            Directory.GetFiles("Assets/PoKingHill/Scenes", "*.unity", SearchOption.AllDirectories);

        [Test]
        public void PhysicsSimulationModeIsScript()
        {
            Assert.AreEqual(SimulationMode.Script, Physics.simulationMode, "PhysX must never auto-step");
            Assert.AreEqual(0.002f, Time.fixedDeltaTime, 1e-7f, "Fixed Timestep must equal training sim_dt");
        }

        [Test, TestCaseSource(nameof(Scenes))]
        public void SceneHasNoPhysXComponents(string scenePath)
        {
            var scene = EditorSceneManager.OpenScene(scenePath, OpenSceneMode.Additive);
            try
            {
                int colliders = 0, rigidbodies = 0, joints = 0, controllers = 0;
                foreach (var root in scene.GetRootGameObjects())
                {
                    colliders += root.GetComponentsInChildren<Collider>(true).Length;
                    rigidbodies += root.GetComponentsInChildren<Rigidbody>(true).Length;
                    joints += root.GetComponentsInChildren<Joint>(true).Length;
                    controllers += root.GetComponentsInChildren<CharacterController>(true).Length;
                }
                Assert.Zero(colliders, $"{scenePath}: Collider components present");
                Assert.Zero(rigidbodies, $"{scenePath}: Rigidbody components present");
                Assert.Zero(joints, $"{scenePath}: PhysX Joint components present");
                Assert.Zero(controllers, $"{scenePath}: CharacterController present");
            }
            finally { EditorSceneManager.CloseScene(scene, true); }
        }
    }
}
