// PostsContext.js
import { createContext, useState, useEffect } from "react";

export const PostsContext = createContext();

export const PostsProvider = ({ children }) => {
  // Load initial posts from localStorage
  const [posts, setPosts] = useState(() => {
    const saved = localStorage.getItem("deepfake_posts");
    return saved ? JSON.parse(saved) : [];
  });

  // Save to localStorage whenever posts change
  useEffect(() => {
    try {
      localStorage.setItem("deepfake_posts", JSON.stringify(posts));
    } catch (error) {
      console.error("Local storage save failed:", error);
      // If it's a QuotaExceededError, we could potentially alert the user or clear some logs
      // but at least we don't crash the whole app session.
    }
  }, [posts]);

  const updateLikes = (postId) => {
    setPosts((prevPosts) =>
      prevPosts.map((post) =>
        post.id === postId ? { ...post, likes: post.likes === 1 ? 0 : 1 } : post
      )
    );
  };

  const addPost = (newPost) => {
    // Ensure post has a unique ID and initial likes if not present
    const postWithId = {
      ...newPost,
      id: newPost.id || Date.now(),
      likes: newPost.likes || 0
    };
    setPosts((prevPosts) => [...prevPosts, postWithId]);
  };

  const deletePost = (postId) => {
    setPosts((prevPosts) => prevPosts.filter((post) => post.id !== postId));
  };

  return (
    <PostsContext.Provider value={{ posts, addPost, updateLikes, deletePost }}>
      {children}
    </PostsContext.Provider>
  );
};

