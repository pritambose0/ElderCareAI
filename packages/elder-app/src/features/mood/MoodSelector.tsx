import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Smile, Meh, Frown, Sparkles } from "lucide-react";
import { CameraMoodDetector } from "./CameraMoodDetector";
import useAICompanion from "@/hooks/useAICompanion";
import { motion, AnimatePresence } from "framer-motion";

export const MoodSelector = () => {
    // AI / Camera State
    const { detectMoodFromImage, detectedImageMood } = useAICompanion({ elderId: "elder-demo", autoConnect: true });
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [feedback, setFeedback] = useState<string | null>(null);

    // Manual / DB State
    const [saving, setSaving] = useState(false);
    const [lastMood, setLastMood] = useState<string | null>(null);

    // AI Detection Handler
    const handleCapture = (imageData: string) => {
        setIsAnalyzing(true);
        detectMoodFromImage(imageData);
    };

    // Function to handle saving mood (reused by both manual and AI)
    const saveMoodToDB = async (mood: string, source: 'manual' | 'camera') => {
        setSaving(true);
        try {
            const { auth, db } = await import("@elder-nest/shared");
            const { collection, addDoc, serverTimestamp } = await import("firebase/firestore");

            const user = auth.currentUser;
            if (!user) {
                console.warn("No user logged in, cannot save mood");
                return;
            }

            // Save mood to Firestore
            await addDoc(collection(db, "moods"), {
                userId: user.uid,
                score: mood.toLowerCase().includes('happy') ? 1.0 : mood.toLowerCase().includes('sad') ? 0.0 : 0.5,
                label: mood,
                source: source,
                aiFeedback: source === 'camera' ? detectedImageMood : null,
                timestamp: serverTimestamp()
            });

        } catch (error) {
            console.error("Failed to save mood", error);
        } finally {
            setSaving(false);
        }
    };

    // Watch for AI results and auto-select/save
    useEffect(() => {
        if (detectedImageMood) {
            setIsAnalyzing(false);
            setFeedback(detectedImageMood);

            // Map AI mood to standard moods for saving logic
            const lowerMood = detectedImageMood.toLowerCase();
            let mappedMood = 'okay';
            if (lowerMood.includes('happy') || lowerMood.includes('joy')) {
                mappedMood = 'happy';
            } else if (lowerMood.includes('sad') || lowerMood.includes('angry')) {
                mappedMood = 'sad';
            }

            // Save the AI detected mood
            saveMoodToDB(mappedMood, 'camera');

            // Auto hide feedback after 5 seconds
            const timer = setTimeout(() => {
                setFeedback(null);
            }, 5000);
            return () => clearTimeout(timer);
        }
    }, [detectedImageMood]);

    const handleMoodClick = async (mood: string) => {
        setLastMood(mood);
        await saveMoodToDB(mood, 'manual');

        // Reset selection visual after 3 seconds
        setTimeout(() => setLastMood(null), 3000);
    };

    return (
        <div className="space-y-6 w-full">
            {/* Manual Mood Buttons */}
            <div className="flex gap-4 justify-center w-full">
                <Button
                    variant={lastMood === 'happy' ? 'default' : 'outline'}
                    size="xl"
                    className={`flex-1 flex-col gap-2 h-auto py-8 transition-all rounded-3xl ${lastMood === 'happy'
                            ? 'bg-green-500 text-white border-green-600 scale-105 shadow-xl dark:bg-green-600'
                            : 'bg-white hover:bg-green-50 text-slate-700 border-2 border-slate-200 dark:bg-slate-800 dark:text-slate-200 dark:border-slate-700 dark:hover:bg-slate-700 dark:hover:border-green-500'
                        }`}
                    onClick={() => handleMoodClick('happy')}
                    disabled={saving}
                >
                    <Smile size={48} className={lastMood === 'happy' ? 'text-white' : 'text-green-500'} />
                    <span className="text-xl font-bold">Happy</span>
                </Button>

                <Button
                    variant={lastMood === 'okay' ? 'default' : 'outline'}
                    size="xl"
                    className={`flex-1 flex-col gap-2 h-auto py-8 transition-all rounded-3xl ${lastMood === 'okay'
                            ? 'bg-yellow-400 text-slate-900 border-yellow-500 scale-105 shadow-xl dark:bg-yellow-500'
                            : 'bg-white hover:bg-yellow-50 text-slate-700 border-2 border-slate-200 dark:bg-slate-800 dark:text-slate-200 dark:border-slate-700 dark:hover:bg-slate-700 dark:hover:border-yellow-400'
                        }`}
                    onClick={() => handleMoodClick('okay')}
                    disabled={saving}
                >
                    <Meh size={48} className={lastMood === 'okay' ? 'text-slate-900' : 'text-yellow-500'} />
                    <span className="text-xl font-bold">Okay</span>
                </Button>

                <Button
                    variant={lastMood === 'sad' ? 'default' : 'outline'}
                    size="xl"
                    className={`flex-1 flex-col gap-2 h-auto py-8 transition-all rounded-3xl ${lastMood === 'sad'
                            ? 'bg-red-500 text-white border-red-600 scale-105 shadow-xl dark:bg-red-600'
                            : 'bg-white hover:bg-red-50 text-slate-700 border-2 border-slate-200 dark:bg-slate-800 dark:text-slate-200 dark:border-slate-700 dark:hover:bg-slate-700 dark:hover:border-red-500'
                        }`}
                    onClick={() => handleMoodClick('sad')}
                    disabled={saving}
                >
                    <Frown size={48} className={lastMood === 'sad' ? 'text-white' : 'text-red-500'} />
                    <span className="text-xl font-bold">Sad</span>
                </Button>
            </div>

            {/* AI Camera Section */}
            <div className="flex justify-center">
                <div className="w-full max-w-sm">
                    <CameraMoodDetector
                        onCapture={handleCapture}
                        isAnalyzing={isAnalyzing}
                    />
                </div>
            </div>

            {/* AI Feedback Area */}
            <AnimatePresence>
                {(feedback || isAnalyzing) && (
                    <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: "auto" }}
                        exit={{ opacity: 0, height: 0 }}
                        className="bg-indigo-50 dark:bg-indigo-900/30 p-4 rounded-xl border border-indigo-100 dark:border-indigo-800 text-center"
                    >
                        {isAnalyzing ? (
                            <p className="text-indigo-600 dark:text-indigo-300 animate-pulse">
                                Analyzing your beautiful smile...
                            </p>
                        ) : (
                            <div className="flex flex-col items-center gap-2">
                                <Sparkles className="text-amber-400 h-6 w-6" />
                                <p className="text-lg font-medium text-indigo-900 dark:text-indigo-100">
                                    I think you look <span className="font-bold text-indigo-600 dark:text-indigo-300 capitalize">{feedback}</span>!
                                </p>
                            </div>
                        )}
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    )
}
